import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Polygon
from tqdm import tqdm
import seaborn as sns
import numpy as np
from math import atan2, degrees


from settings import CONSTANTS

def create_subset(log_center, lat_center, size_meters, label, size_entries_limit = 1e5):
    """
    This method creates a subset of entries centered around a given point.
    """

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

    return data

def get_area_center(data):
    """
    This method returns the long, lat center for the data (as mean of the extremes).
    """

    return (data["longitude"].max() + data["longitude"].min()) / 2, (data["latitude"].max() + data["latitude"].min()) / 2


def calculate_polygon_size(polygon, cell_size_in_degrees=CONSTANTS.DEFAULT_CELL_SIZE_METERS * CONSTANTS.METERS_TO_DEGREES):
    # Calculate the size of the polygon in terms of the cell size
    min_x, min_y, max_x, max_y = polygon.bounds
    width = (max_x - min_x) / cell_size_in_degrees
    height = (max_y - min_y) / cell_size_in_degrees
    return max(width,height)/2


def add_derivate_columns(data, cell_size_in_degrees=CONSTANTS.DEFAULT_CELL_SIZE_METERS * CONSTANTS.METERS_TO_DEGREES):

    # We will create a naive long and lat coordinates relative to the center of the area that we want to analyze, this allows us to make easy the math to locate the corresponding cell with just a simple division.
    long_area_center, lat_area_center = get_area_center(data)
    

    data["relative_lat"] = data["latitude"] - lat_area_center
    data["relative_lon"] = data["longitude"] - long_area_center

    # Now we could assign each building center to a coordinate pair on the grid just dividing the relative coordinates by the cell size in degrees and rounding down to the nearest integer
    data["cell_long_pos"] = data["relative_lon"].floordiv(cell_size_in_degrees).astype(int)
    data["cell_lat_pos"] = data["relative_lat"].floordiv(cell_size_in_degrees).astype(int)

    # We still need to know how much close cells could overlap with the building, so we will define the building size in terms of the cell size.
    data["size_in_cells"] = data["geometry"].apply(calculate_polygon_size).astype(int) + 1 # We make +1 because we need to include at least one cell on each side in case that the building is close to the cell border. For example, in the case that the size (that is the max radius) is 0.79 cells, that means that we need to check all contiguous cells. In the case that the radius is 1.2 we need to check at least 2 cells on each side (and corners) because if the building center is very close to the cell border it could cross an entire cell on the side and reach the next one. 

    return data

def create_polygon(x, y, cell_size_in_degrees, reference_x = 0, reference_y = 0):
    # Create a polygon for each cell
    coords = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
    coords = [(coord[0] * cell_size_in_degrees + reference_x, coord[1] * cell_size_in_degrees + reference_y) for coord in coords]
    poly = Polygon(coords)
    return poly

def create_grid(data, cell_size_in_degrees=CONSTANTS.DEFAULT_CELL_SIZE_METERS * CONSTANTS.METERS_TO_DEGREES):
    
    x_min = data["cell_long_pos"].min()
    x_max = data["cell_long_pos"].max()
    y_min = data["cell_lat_pos"].min()
    y_max = data["cell_lat_pos"].max()
    reference_x, reference_y = get_area_center(data)

    # Create a grid of polygons
    x_coords = list(range(x_min, x_max + 1))
    y_coords = list(range(y_min, y_max + 1))
    grid_polygons = {}
    for x in tqdm(x_coords):
        grid_polygons[x] = {}
        for y in y_coords:
            # Create a polygon for each cell
            poly = create_polygon(x, y, cell_size_in_degrees, reference_x, reference_y)
            grid_polygons[x][y] = poly

    return grid_polygons

def add_overlapping_cells(data, grid_polygons):
    data["overlapping"] = data.progress_apply(lambda row: found_overlapping_cells(row, grid_polygons), axis=1)
    return data

def found_overlapping_cells(row, grid_polygons):
    # Get the cell coordinates
    cell_x = row["cell_long_pos"]
    cell_y = row["cell_lat_pos"]
    cells_size = row["size_in_cells"]
    # Get the polygon
    polygon = row["geometry"]
    # Create a list to store the overlapping cells
    overlapping = []
    cell_polygon = grid_polygons[cell_x][cell_y]
    # Check if the polygon is completely inside the cell
    if polygon.intersects(cell_polygon):
        if abs(polygon.intersection(cell_polygon).area - polygon.area) < 0.0001 * polygon.area: # We can use a tolerance to check if the polygon is completely inside the cell
            overlapping.append({"cell_long_pos": cell_x, "cell_lat_pos": cell_y, "area": polygon.area, "polygon_tag": row["full_plus_code"]})
            return overlapping
    # Check the surrounding cells
    for i in range(-cells_size, cells_size + 1):
        for j in range(-cells_size, cells_size + 1):
            target_cell_x = cell_x + i
            target_cell_y = cell_y + j
            # Check if the target cell is within the grid
            if target_cell_x in grid_polygons and target_cell_y in grid_polygons[target_cell_x]:
                cell_polygon = grid_polygons[target_cell_x][target_cell_y]
                if polygon.intersects(cell_polygon):
                    overlapping_area = polygon.intersection(cell_polygon)
                    overlapping.append({"cell_long_pos": target_cell_x, "cell_lat_pos": target_cell_y, "area": overlapping_area.area, "polygon_tag": row["full_plus_code"]})
                    if overlapping_area == polygon.area:
                        # If the polygon is completely inside the cell, we can skip it
                        return overlapping
    return overlapping

def build_intersections_df(data):

    intersections = data["overlapping"].explode().reset_index(drop=True)
    intersections = pd.DataFrame(intersections.tolist())
    intersections["relative_weight"] = intersections["area"] / intersections.groupby(["cell_long_pos", "cell_lat_pos"])["area"].transform("sum")
    
    return intersections

def plot_occupied_area_heatmap(intersections):
    
    occupied_area = intersections.groupby(["cell_long_pos", "cell_lat_pos"])["area"].sum().reset_index()
    occupied_area_heatmap = occupied_area.pivot_table(index="cell_lat_pos", columns="cell_long_pos", values="area", aggfunc="sum", fill_value=0)
    occupied_area_heatmap.sort_index(ascending=False, inplace=True)
    sns.heatmap(occupied_area_heatmap, cmap="coolwarm", cbar_kws={'label': 'Area Occupied'})

def get_polygon_orientation(polygon, include_eccentricity=True):
    coords = np.array(polygon.exterior.coords[:])
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


def get_orientation_vector(polygon):
    coords = np.array(polygon.exterior.coords)
    coords -= coords.mean(axis=0)
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    i = np.argmax(eigvals)
    direction = eigvecs[:, i]
    return coords.mean(axis=0), direction, eigvals[i]