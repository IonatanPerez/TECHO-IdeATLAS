import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Polygon
from tqdm import tqdm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, degrees
from matplotlib.collections import LineCollection
import os
from shapely.geometry import Point

from settings import CONSTANTS

def create_subset(log_center, lat_center, size_meters, label, size_entries_limit = 1e5):
    """
    This method creates a subset of entries centered around a given point.
    """

    # We check that the file exists
    assert os.path.isfile(CONSTANTS.DATA_FOLDER + CONSTANTS.COMPLETE_DATA_FILE), f"File {CONSTANTS.DATA_FOLDER + CONSTANTS.COMPLETE_DATA_FILE} does not exist. This file must be extracted from the original raw data available on https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_4_gzip/95b_buildings.csv.gz and put into your local data folder that is not sync with git."
    data = pd.read_csv(CONSTANTS.DATA_FOLDER + CONSTANTS.COMPLETE_DATA_FILE)

    lat_max = data["latitude"].max()
    lat_min = data["latitude"].min()
    lon_max = data["longitude"].max()
    lon_min = data["longitude"].min()

    assert log_center > lon_min and log_center < lon_max, "Longitude center is out of bounds"
    assert lat_center > lat_min and lat_center < lat_max, "Latitude center is out of bounds"

    # Convert size from meters to degrees
    size_degrees = size_meters * CONSTANTS.METERS_TO_DEGREES
    # Calculate the bounds of the subset
    lat_min_boundary = lat_center - size_degrees / 2
    lat_max_boundary = lat_center + size_degrees / 2
    lon_min_boundary = log_center - size_degrees / 2
    lon_max_boundary = log_center + size_degrees / 2

    # Filter the data to only include entries within the bounds
    data = data[
        (data["latitude"] >= lat_min_boundary) &
        (data["latitude"] <= lat_max_boundary) &
        (data["longitude"] >= lon_min_boundary) &
        (data["longitude"] <= lon_max_boundary)
    ]

    # We check that the number of entries is not greater than the limit
    assert len(data) < size_entries_limit, "The number of entries is greater than the limit set to avoid performance issues"
    
    # We store the subset in a csv file
    data.to_csv(CONSTANTS.SUBSETS_FOLDER + label + ".csv", index=False)
    print(f"Subset {label} created with {len(data)} entries.")

    return data

def load_subset(label):
    """
    This method loads a subset of entries from a csv file.
    """
    if not label in available_subsets():
        raise ValueError(f"Subset {label} not found. Available subsets: {available_subsets()}")

    data = pd.read_csv(CONSTANTS.SUBSETS_FOLDER + label + ".csv")
    print(f"Subset {label} loaded with {len(data)} entries.")

    return data

def available_subsets():
    """
    This method returns a list of available subsets.
    """

    import os

    files = os.listdir(CONSTANTS.SUBSETS_FOLDER)
    subsets = [f.split(".")[0] for f in files if f.endswith(".csv")]

    return subsets

def convert_to_gpd(data):
    """
    This method preprocesses the data converting dtypes and creating the geopandas dataframe.
    """

    # Convert the data to a GeoDataFrame
    data = data.astype({
        "latitude": "float64",
        "longitude": "float64",
        "confidence": "float64",
        "area_in_meters": "float64",
        "full_plus_code": "string",
        "geometry": "string",
    })

    # Create a GeoDataFrame from the data
    data = gpd.GeoDataFrame(data, geometry=data["geometry"].apply(wkt.loads))
    if data.crs is None:
        data.set_crs(epsg=4326, inplace=True)

    return data

def get_region_centroid(gpf):
    """
    This method returns the centroid of the region.
    """
    # Get the centroid of the region
    centroid = gpf.unary_union.centroid
    return centroid.x, centroid.y

def convert_to_UTM(gdf):
    long, lat = get_region_centroid(gdf)
    zone = int((long + 180) / 6) + 1
    is_northern = lat >= 0
    epsg_code = 32600 + zone if is_northern else 32700 + zone
    points = [Point(xy) for xy in zip(gdf['longitude'], gdf['latitude'])]
    points_gdf = gpd.GeoDataFrame(geometry=points)
    points_gdf.crs = "EPSG:4326"
    points_gdf = points_gdf.to_crs(epsg=epsg_code)
    gdf_utm = gdf.to_crs(epsg=epsg_code)
    gdf_utm["coord_x"] = points_gdf["geometry"].x
    gdf_utm["coord_y"] = points_gdf["geometry"].y
    return gdf_utm

def calculate_polygon_size(polygon):
    min_x, min_y, max_x, max_y = polygon.bounds
    width = (max_x - min_x)
    height = (max_y - min_y)
    return max(width,height)/2


def add_derivate_columns(data):
    # We will create a naive long and lat coordinates relative to the center of the area that we want to analyze, this allows us to make easy the math to locate the corresponding cell with just a simple division.
    area_center_x, area_center_y = get_region_centroid(data)
    

    data["relative_x"] = data["coord_x"] - area_center_x
    data["relative_y"] = data["coord_y"] - area_center_y

    # Now we could assign each building center to a coordinate pair on the grid just dividing the relative coordinates by the cell size in degrees and rounding down to the nearest integer
    data["x_cell"] = data["relative_x"].floordiv(CONSTANTS.CELL_SIZE_METERS).astype(int)
    data["y_cell"] = data["relative_y"].floordiv(CONSTANTS.CELL_SIZE_METERS).astype(int)

    # We still need to know how much close cells could overlap with the building, so we will define the building size in terms of the cell size.
    data["size_in_cells"] = data["geometry"].apply(lambda poly: calculate_polygon_size(poly)/CONSTANTS.CELL_SIZE_METERS).astype(int) + 1 # We make +1 because we need to include at least one cell on each side in case that the building is close to the cell border. For example, in the case that the size (that is the max radius) is 0.79 cells, that means that we need to check all contiguous cells. In the case that the radius is 1.2 we need to check at least 2 cells on each side (and corners) because if the building center is very close to the cell border it could cross an entire cell on the side and reach the next one. 

    return data

def create_polygon(x, y, area_center_x, area_center_y):
    # Create a polygon for each cell
    coords = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
    coords = [(coord[0] * CONSTANTS.CELL_SIZE_METERS + area_center_x, coord[1] * CONSTANTS.CELL_SIZE_METERS + area_center_y) for coord in coords]
    poly = Polygon(coords)
    return poly

def create_grid(data):
    x_min = data["x_cell"].min()
    x_max = data["x_cell"].max()
    y_min = data["y_cell"].min()
    y_max = data["y_cell"].max()
    area_center_x, area_center_y = get_region_centroid(data)

    # Create a grid of polygons
    x_coords = list(range(x_min, x_max + 1))
    y_coords = list(range(y_min, y_max + 1))
    grid_polygons = {}
    for x in tqdm(x_coords):
        grid_polygons[x] = {}
        for y in y_coords:
            # Create a polygon for each cell
            poly = create_polygon(x, y, area_center_x, area_center_y)
            grid_polygons[x][y] = poly

    return grid_polygons

def add_overlapping_cells(data, grid_polygons):
    data["overlapping"] = data.progress_apply(lambda row: found_overlapping_cells(row, grid_polygons), axis=1)
    return data

def found_overlapping_cells(row, grid_polygons):
    # Get the cell coordinates
    x_cell = row["x_cell"]
    y_cell = row["y_cell"]
    cells_size = row["size_in_cells"]
    # Get the polygon
    polygon = row["geometry"]
    # Create a list to store the overlapping cells
    overlapping = []
    cell_polygon = grid_polygons[x_cell][y_cell]
    # Check if the polygon is completely inside the cell
    if polygon.intersects(cell_polygon):
        if abs(polygon.intersection(cell_polygon).area - polygon.area) < 0.0001 * polygon.area: # We can use a tolerance to check if the polygon is completely inside the cell
            overlapping.append({"x_cell": x_cell, "y_cell": y_cell, "area": polygon.area, "polygon_tag": row["full_plus_code"], "fraction_of_the_building": 1})
            return overlapping
    # Check the surrounding cells
    for i in range(-cells_size, cells_size + 1):
        for j in range(-cells_size, cells_size + 1):
            target_cell_x = x_cell + i
            target_cell_y = y_cell + j
            # Check if the target cell is within the grid
            if target_cell_x in grid_polygons and target_cell_y in grid_polygons[target_cell_x]:
                cell_polygon = grid_polygons[target_cell_x][target_cell_y]
                if polygon.intersects(cell_polygon):
                    overlapping_area = polygon.intersection(cell_polygon)
                    overlapping.append({"x_cell": target_cell_x, "y_cell": target_cell_y, "area": overlapping_area.area, "polygon_tag": row["full_plus_code"], "fraction_of_the_building": overlapping_area.area / polygon.area})
                    if overlapping_area == polygon.area:
                        # If the polygon is completely inside the cell, we can skip it
                        return overlapping
    return overlapping

def build_intersections_df(data):

    intersections = data["overlapping"].explode().reset_index(drop=True)
    intersections = pd.DataFrame(intersections.tolist())
    intersections["fraction_of_buildings_in_cell"] = intersections["area"] / intersections.groupby(["x_cell", "y_cell"])["area"].transform("sum")
    
    return intersections

def build_cell_composition(intersections):
    # We will create a dataframe with the cell composition
    cell_composition = intersections.groupby(["x_cell", "y_cell"])[["polygon_tag", "fraction_of_buildings_in_cell", "fraction_of_the_building", "area"]].apply(
        lambda x: [{"polygon_tag": row["polygon_tag"], "fraction_of_buildings_in_cell": row["fraction_of_buildings_in_cell"], "fraction_of_the_building": row["fraction_of_the_building"], "area": row["area"]} for _, row in x.iterrows()]
    )
    cell_composition = cell_composition.to_frame()
    cell_composition.columns = ["cell_composition"]
    return cell_composition

def plot_occupied_area_heatmap(intersections, area_center_x, area_center_y, save_as = None):
    occupied_area = intersections.groupby(["x_cell", "y_cell"])["area"].sum().reset_index()
    occupied_area["cell_occupancy_fraction"] = occupied_area["area"] / (CONSTANTS.CELL_SIZE_METERS) ** 2
    occupied_area_heatmap = occupied_area.pivot_table(index="y_cell", columns="x_cell", values="cell_occupancy_fraction", aggfunc="sum", fill_value=0)
    # We want to recover the original cell coordinates
    index = occupied_area_heatmap.index * CONSTANTS.CELL_SIZE_METERS + area_center_y
    columns = occupied_area_heatmap.columns * CONSTANTS.CELL_SIZE_METERS + area_center_x
    occupied_area_heatmap.index = index
    occupied_area_heatmap.columns = columns
    occupied_area_heatmap.sort_index(ascending=False, inplace=True)
    sns.heatmap(occupied_area_heatmap, cmap="coolwarm", cbar_kws={'label': 'Fraction of area occupied'})
    
    plt.xlabel("West-East")
    plt.ylabel("South-North")
    plt.title("Cell occupancy fraction")
    # Show only a few rounded ticks on axes, set them horizontal
    plt.xticks(
        ticks=np.round(np.linspace(plt.gca().get_xticks()[0], plt.gca().get_xticks()[-1], 6), 5),
        rotation=0
    )
    plt.yticks(
        ticks=np.round(np.linspace(plt.gca().get_yticks()[0], plt.gca().get_yticks()[-1], 6), 5),
        rotation=0
    )
    if save_as is not None:
        plt.tight_layout()
        plt.savefig(save_as)
    plt.show()

def get_polygon_orientation(polygon, include_eccentricity=True):
    mrr = polygon.minimum_rotated_rectangle # With this we ensure the independence of the number of points density of the polygon
    coords = np.array(mrr.exterior.coords[:])
    if (coords[0] == coords[-1]).all():
        coords = coords[:-1] # Exclude the last point to avoid duplication because closing polygon point that bias the results
    coords -= coords.mean(axis=0)
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    angle_rad = atan2(principal_axis[1], principal_axis[0])
    angle_deg = degrees(angle_rad)
    eigvals = np.sort(eigvals)[::-1]
    if include_eccentricity:
        eccentricity = np.sqrt(1 - (eigvals[1] / eigvals[0]))
        return angle_deg % 180, eccentricity
    else:
        return angle_deg % 180
    
def add_building_orientation(gdf):
    # Calculate the orientation for each polygon
    gdf["orientation"] = gdf.progress_apply(lambda row: get_polygon_orientation(row["geometry"]), axis=1)
    # Split the orientation into two columns
    gdf["orientation_angle"] = gdf["orientation"].apply(lambda x: x[0])
    gdf["eccentricity"] = gdf["orientation"].apply(lambda x: x[1])
    gdf.drop(columns=["orientation"], inplace=True)
    return gdf
    
def get_orientation_for_many_polygons(polygons, weights = None, include_eccentricity=True):
    if weights is None:
        weights = np.ones(len(polygons))
    assert len(polygons) == len(weights), "The number of polygons and weights must be the same"
    all_coords = []
    for i, polygon in enumerate(polygons):
        mrr = polygon.minimum_rotated_rectangle # With this we ensure that all polygons had the same weight
        coords = np.array(mrr.exterior.coords[:])
        if (coords[0] == coords[-1]).all():
            coords = coords[:-1]
        coords -= coords.mean(axis=0)
        coords = coords * np.sqrt(weights[i]) # We scale the coordinates by the square root of the weight to ensure that the weight is proportional to the area
        all_coords.append(coords)
    all_coords = np.concatenate(all_coords)
    cov = np.cov(all_coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    angle_rad = atan2(principal_axis[1], principal_axis[0])
    angle_deg = degrees(angle_rad)
    eigvals = np.sort(eigvals)[::-1]
    if include_eccentricity:
        eccentricity = np.sqrt(1 - (eigvals[1] / eigvals[0]))
        return angle_deg % 180, eccentricity
    else:
        return angle_deg % 180

def calculate_orientation_for_cell(cell_composition_row, buildings_df):
    # Get the cell composition
    cell_composition = cell_composition_row["cell_composition"]
    # Get the buildings in the cell
    buildings = []
    for building in cell_composition:
        building_id = building["polygon_tag"]
        building_fraction = building["fraction_of_the_building"]
        # Get the building from the dataframe
        building_df = buildings_df[buildings_df["full_plus_code"] == building_id]
        building_polygon = building_df.iloc[0]["geometry"]
        buildings.append({"polygon": building_polygon, "weight": building_fraction})
    # Calculate the orientation for each polygon
    orientations = get_orientation_for_many_polygons([building["polygon"] for building in buildings], [building["weight"] for building in buildings])
    return orientations

def add_orientation_to_cells(composition, buildings_df):
    # Calculate the orientation for each cell
    composition["orientation"] = composition.progress_apply(lambda row: calculate_orientation_for_cell(row, buildings_df), axis=1)
    # Split the orientation into two columns
    composition["orientation_angle"] = composition["orientation"].apply(lambda x: x[0])
    composition["eccentricity"] = composition["orientation"].apply(lambda x: x[1])
    composition.drop(columns=["orientation"], inplace=True)
    return composition

def plot_orientation_lines(cell_composition, area_center_x=0, area_center_y=0, save_as=None):
    """
    Dibuja una línea centrada en cada celda, con orientación determinada por el ángulo
    y longitud proporcional a la excentricidad. Robusto a la escala.
    """
    df = cell_composition.reset_index().copy()

    df["center_x"] = df["x_cell"] * CONSTANTS.CELL_SIZE_METERS + area_center_x + CONSTANTS.CELL_SIZE_METERS / 2
    df["center_y"] = df["y_cell"] * CONSTANTS.CELL_SIZE_METERS + area_center_y + CONSTANTS.CELL_SIZE_METERS / 2

    df["angle_rad"] = np.deg2rad(df["orientation_angle"])

    max_length = 0.7 * CONSTANTS.CELL_SIZE_METERS
    df["length"] = df["eccentricity"] * max_length

    segments = []
    for _, row in df.iterrows():
        dx = np.cos(row["angle_rad"]) * row["length"] / 2
        dy = np.sin(row["angle_rad"]) * row["length"] / 2
        x0, y0 = row["center_x"], row["center_y"]
        segment = [(x0 - dx, y0 - dy), (x0 + dx, y0 + dy)]
        segments.append(segment)

    fig, ax = plt.subplots(figsize=(12, 10))
    line_collection = LineCollection(segments, colors="black", linewidths=1.2, alpha=0.8)
    ax.add_collection(line_collection)

    ax.set_xlim(df["center_x"].min() - CONSTANTS.CELL_SIZE_METERS, df["center_x"].max() + CONSTANTS.CELL_SIZE_METERS)
    ax.set_ylim(df["center_y"].min() - CONSTANTS.CELL_SIZE_METERS, df["center_y"].max() + CONSTANTS.CELL_SIZE_METERS)

    ax.set_xlabel("West-East")
    ax.set_ylabel("South-North")
    ax.set_title("Cell Orientation (line direction) and Eccentricity (line length)")
    ax.grid(True, linestyle='--', alpha=0.3)

    if save_as:
        plt.tight_layout()
        plt.savefig(save_as)
    plt.show()