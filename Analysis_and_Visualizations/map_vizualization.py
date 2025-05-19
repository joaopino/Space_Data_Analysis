"""
Auxiliary functions for supporting the visualization of clusters on the Portuguese map.

This module provides helper functions to facilitate the creation, customization, and display of cluster visualizations
over geographic representations of Portugal. The functions handle tasks such map plotting, and annotation to enhance the interpretability of clustering results in a geospatial context.

Intended for use in presentation of clustering results involving Portuguese geographic data divided by municipalities.
"""
import geopandas as gpd


def add_geodata(geojson_path,dataframe):
    """
    Add geospatial data to a DataFrame using a GeoJSON file.

    Parameters
    ----------
    geojson_path : str
        Path to the GeoJSON file containing the geospatial data.
    dataframe : pd.DataFrame
        DataFrame to which the geospatial data will be added.

    Returns
    -------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing the merged data.
    """
    # Load the GeoJSON file
    gdf = gpd.read_file(geojson_path)

    # Merge the GeoDataFrame with the DataFrame based on 'id_municipio'
    gdf = gdf.merge(dataframe, left_on='id_municipio', right_on='id_municipio')

    return gdf

