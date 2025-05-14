# IDEAtlas

On this project we want to build morphological metrics of the buildings according to the information provided by the Open Buildings project of Google for a specific region on Buenos Aires, Argentina. The goal is to create images with a resolution of 10x10 meters that represent the buildings density and orientation of the buildings on the region.

## Data source

The data source is a file available on [Open Building project](https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_4_gzip/95b_buildings.csv.gz). This file contains the polygons of the buildings in the region. Based on this polygons we want to create the images with the morphological metrics. This file need to be fetched and unzipped on each local machine because it can't be stored into the online repository.

## Data processing

In order to process the data we decided to take a subset of all the information available on the file. We created on DataPreparation.ipynb (using tools.py the main list of functions) the option to create smaller CSV files with just the entries that correspond to a specific region around some desired long-lat coordinates. This the only previous step that you need to run previously to be able to run the general pipeline.

### Data preparation

The raw data is very clean and healthy, so it's necessary to make any kind of cleaning. The data itself has a column called "confidence", but we don't used that column because we assume that the provided data is good enough and we have no criteria to filter data by this information.

Each entry has the Polygon with the shape and location of the building (everything on long-lat coordinates) and an ID ("full_plus_code", a system of Google to tag places) that is unique for each building. Also each entry has the area (in meters) of the building (that match pretty good with the area of the polygon based on the coordinates).

In order to make all calculations measured in meters and avoid the problems of shape conservation or distortions due different scales (degrees to meters) on each axis we decided to convert the long-lat coordinates to meters. To do that we used the geopandas included tools library. We converted the input geographic coordinates (WGS 84, EPSG:4326) to projected UTM coordinates (EPSG:326xx/327xx) to enable precise spatial analysis in meters. We could do this because we are working on a small region (less than 100km) and the UTM projection is valid for this kind of regions.

With this data already converted we define a grid size, by default 10x10 meters (but it could be changed on settings) and then the main task is to correlate witch building belows each cell. Due the cell size is similar to the size of the buildings, it's very common that a cell has more than a building but also that a building has parts that belong to different cells.

To avoid make an algorithm that groves O(n^2) with the size of the sample (if we check every cell against every building overlapping) we decide to build the cells on an arbitrary coordinate system where each cell has a unitary numbering system on each direction (north-south ans east-west). So we center the numbering system on the center of the region and make a grid that goes from -N/2 to +N/2 where N is the number of times that the cell size in meters fit on the region. So the $cell_{00}$ is the central cell and the $cell_{-N/2-N/2}$ is the cell that is on the left bottom corner of the region.

Then the question is how to assign the buildings to the corresponding cells. To do that we take the coordinate of the building (we use here the reported one) and we subtract in both axis the coordinates of the center of the region. Then we divide the result by the cell size and round to the nearest minor integer. This way we get the cell coordinates that corresponds to the building.

Then we need to check if the building overlaps with contiguous cells. To do that we take the max edge of the rectangle that include the building Polygon and divide it by 2. This is the "radius" that we need to check around.

Once we have this information for each building we create a dictionary with the cell coordinates and we populate it with square Polygons that represent the cells area.

With both objects we iterate over each building once and we check if the building Polygon overlaps any of the rounding to the building cells, and when there is overlap we add that information to the building entry. So we have the connection between the building and the cells.

With this information we could create a new dataframe that connects cells and buildings and iterate over the cells to check for each cell witch buildings are inside. The needed information to calculate the covered area is already calculated on this stage because is just add all overlaps areas on the cell.

### Orientations calculation

In order to detect the orientation of the buildings we need to determine how to make it. There are many ways, one is to perform a PCA over the point that define the Polygon, other is to search for the small rectangle that contains the Polygon and then calculate the angle of the rectangle (here again we need some kind of PCA). The problem of working with the PCA is that PCA over the original Polygon coordinates could have biases due the fact than complex morphologies have not homogeneous distribution of points, so if we make a PCA over the original coordinates we could get a biases orientation. The most simple realization of this problem is that all closed Polygons have a duplicated point (the first and the last point are the sam, so we need to drop one of them). To avoid this problems we decided to combine both methods. We first search for the small rectangle that contains the Polygon and then we calculate the PCA of the points of the rectangle.

When we calculate the PCA we have another advantage, the main autovector provides the orientation, but also the relationship between the two autovectors provides the elongation (or eccentricity) of the building, so we could weight the orientation by its excentricity.

Once we have each building orientation we need to assign a orientation to each cell (in fact the orientation of each building is not relevant at the end). Here we need to understand and define how we will merge into a metric the orientation of many buildings on the same cell. There is no sense make things as the average of the buildings orientation.

On this case we decided "create" a new "building" that is composed by all the fractions of the buildings that belong to the cell and calculate over all the points that compose this new object the PCA.

As the PCA calculation include move the Polygon to the center of the region the position of each building (or building fraction) is not relevant. What is relevant is how much of each building is on each cell and the orientation of the entire building that fraction belows (we can't just cut the buildings parts that match the cell border).

So, with the list of buildings that belong to the cell we make the following steps:

1 - Searching the small rectangle that contains each Polygon and extract for each of them the coordinates of the rectangle. This is done using the Shapely library that provides a function to calculate the minimum bounding rectangle of a Polygon.
2 - Scale the size of each rectangle by the fraction of the building that belongs to the cell. This is done using the area of the building and the area of the intersection between the building and the cell.
3 - Join all points and calculate the PCA over them. Here we are making the same math than if the building was the union of all the buildings that belong to the cell.

## Outputs

On pipeline we build partial visualizations of the data but also under Outputs folder we store the three required outputs with geojson format.

## Libraries and tools utilized

For general data processing we used mostly Pandas and Geopandas (for transforming the raw data into polygons and make switch of spatial representations). We used Shapely to manage the Polygons objects and check the overlapping areas. We use seaborn to represent the heatmap and matplotlib to represent the orientation of the buildings.