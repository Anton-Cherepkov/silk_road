import geopandas as gpd


def shapefile_to_geojson(
    input: str,
    output: str,
    epsg: int = 4326,
) -> None:
    data = gpd.read_file(input)

    data_reprojected = data.copy()
    data_reprojected['geometry'] = data_reprojected['geometry'].to_crs(epsg=epsg)
    data_reprojected.to_file(output, driver="GeoJSON")
