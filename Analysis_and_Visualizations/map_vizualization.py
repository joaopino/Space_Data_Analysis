"""
Auxiliary functions for supporting the visualization of clusters on the Portuguese map.

This module provides helper functions to facilitate the creation, customization, and display of cluster visualizations
over geographic representations of Portugal. The functions handle tasks such map plotting, and annotation to enhance the interpretability of clustering results in a geospatial context.

Intended for use in presentation of clustering results involving Portuguese geographic data divided by municipalities.
"""
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.io as pio
import pyproj
import os
def analyse_geodata(geojson_path):
    '''
    Function for analysing the geospatial data in the GeoJSON file.
    '''
    gdf = gpd.read_file(geojson_path)
    
    # Debugging
    print("Columns:", gdf.columns.tolist())
    print("CRS:", gdf.crs)
    print("Number of features:", len(gdf))
    print("Geometry type:", gdf.geom_type.unique())
    gdf.head()




def add_geodata(geojson_path,dataset_path):
    """
    Combine a dataset with municipalities and data with a GeoJSON file
    The dataframe must contain a column named 'local' to match with the GeoJSON "Concelho" column.
    The function merges both with the uppercase of the strings.

    Parameters
    ----------
    geojson_path : str
        Path to the GeoJSON file containing the geospatial data.
    dataset_path : str
        Path to the dataset containing the data to be merged.

    Returns
    -------
    merged_df : gpd.GeoDataFrame
        GeoDataFrame containing the merged data.
        
    Prints
    -------
    Differences between the two dataframes.
    """
    
    gdf = gpd.read_file(geojson_path)
    df_dataset = pd.read_csv(dataset_path)
    
    # Rename 'municipio' column to 'Concelho' for the case of the portuguese islands
    if ("Municipio" in gdf.columns):
        gdf = gdf.rename(columns={"Municipio": "Concelho"})
         
    elif ("Municipio".upper() in gdf.columns):
        gdf = gdf.rename(columns={"Municipio".upper(): "Concelho"})
        
    
    #Check if collumns necesary for merging exist
    if "local" not in df_dataset.columns:
        print("Warning: The dataset does not contain a 'local' column. Exiting.")
        return
    elif "Concelho" not in gdf.columns:
        print("Warning: The dataset does not contain a 'Concelho' column. Exiting.")
        return
    
        
    # Merge the GeoDataFrame with the DataFrame based on 'id_municipio'
    merged_df = pd.merge(
        gdf,
        df_dataset,
        left_on=gdf['Concelho'].str.upper(),
        right_on=df_dataset['local'].str.upper(),
        how='inner'
    )
    
    
    ##Check the commons values and the ones left out
    df_geodata = gdf['Concelho'].str.upper()
    df_clusters = df_dataset['local'].str.upper()
    common_values = set(df_geodata.str.upper()).intersection(set(df_clusters.str.upper()))
    print("Number of common values:", len(common_values))
    # print("Common values:", common_values)

    not_in_geodata = set(df_clusters.str.upper()) - common_values
    not_in_clusters = set(df_geodata.str.upper()) - common_values

    print("Values in dataset but not in geojson:", not_in_geodata)
    print("Values in geojson but not in dataset:", not_in_clusters) ##Left out shoudld be in Azores or Madeira
        

    return merged_df



def plot_map(dataframe, attribute, color_map):
    '''

        Plot a map of Portugal that highlights specific clusters of the given attribute.
        The maps is generated, ploted and saved as a PNG file on the folder cluster_maps/
        
        Parameters
        ----------
        dataframe : gpd.GeoDataFrame
            The GeoDataFrame containing the geospatial data and the data added from the original dataset.
            It should contain a column with the name of the attribute to be visualized as well as the collumn "CONCELHO".
        attribute : str
            The name of the attribute/column in the GeoDataFrame to be highlighted on the map.
            The data should be numeric and have the values in the color map.
        title : str
            The title of the plot. 
        color_map : str
            The color map to be used for the visualization. 
        
        Returns
        -------
        None
            The function saves the plot as a PNG file in the specified directory.
        Prints
        -------
            
                The function does not return any value, but it saves the plot as a PNG file.

    '''
    dataframe.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    dataframe[attribute] = dataframe[attribute].astype(str) #Sees the attribute as a caregory instead of a scale

    
    fig = px.choropleth(
        dataframe, 
        geojson=dataframe.geometry,    
        locations=dataframe.index,
        color=attribute,
        color_discrete_map=color_map,
        hover_data={
            "Concelho": True,    # show
            attribute: True,       # show
        }
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>" +
                  "Cluster: %{customdata[1]}<extra></extra>")
    
    # Ensure the color scale is discrete and add a legend title
    fig.update_layout(
        legend_title_text=attribute,
        coloraxis_showscale=False,
    )
    
    fig.show()
    
    
    
def saves_pic_of_clusters_analysis(dataframe, attribute, color_map, title, output_folder):

    dataframe.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    dataframe[attribute] = dataframe[attribute].astype(str) #Sees the attribute as a caregory instead of a scale

    
    fig = px.choropleth(
        dataframe, 
        geojson=dataframe.geometry,    
        locations=dataframe.index,
        color=attribute,
        color_discrete_map=color_map,
        hover_data={
            "Concelho": True,    # show
            attribute: True,       # show
        }
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>" +
                  "Cluster: %{customdata[1]}<extra></extra>")
    
    # Ensure the color scale is discrete and add a legend title
    fig.update_layout(
        legend_title_text=attribute,
        coloraxis_showscale=False,
    )
    
    # Save the plot as a PNG file in the cluster_maps/ directory

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{title.replace(' ', '_')}.png")
    pio.write_image(fig, output_path, format='png')
    print(f"Plot saved as {output_path}")
   
    

