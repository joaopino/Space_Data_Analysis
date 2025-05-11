import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px



def processar_ficheiro(caminho_ficheiro, header=4, pasta_saida='.', padding_left=0, padding_right=0):
    '''
    Processes an Excel file and returns a DataFrame.
    
    Parameters:
    caminho_ficheiro (str): Path to the Excel file.
    header (int): Row number to use as the column names.
    pasta_saida (str): Directory to save the processed file.
    padding_left (int): Number of columns to skip on the left. Use in case of an empty/unnecessary column on the left.
    padding_right (int): Number of columns to remove from the right. Use to trim unwanted trailing columns.
    
    Returns:
    df (DataFrame): Processed DataFrame.
    '''

    nome_base = os.path.splitext(os.path.basename(caminho_ficheiro))[0]
    
    df = pd.read_excel(caminho_ficheiro, header=header)
    df.columns = df.columns.map(str).str.strip()

    # Remove left-padding columns
    if padding_left > 0:
        df = df.iloc[:, padding_left:]

    # Remove right-padding columns
    if padding_right > 0:
        df = df.iloc[:, :-padding_right]

    # Rename first columns if present
    if len(df.columns) >= 2:
        df.columns.values[0] = 'local'
        df.columns.values[1] = 'codigo'

    # Drop unnamed columns and rows that are fully NaN (except the first column)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df = df[~df.iloc[:, 1:].isna().all(axis=1)]
    df = df.fillna(0)

    # Remove rows where both the first and second columns are NaN    
    df = df[~(df.iloc[:, 0].isna() & df.iloc[:, 1].isna())]

    # Optional saving to file (commented)
    # caminho_csv = os.path.join(pasta_saida, f'{nome_base}.csv')
    # df.to_csv(caminho_csv, index=False)

    return df

def filter_for_only_municipalities(df):
    '''
    Filters the DataFrame to include only rows where the 'codigo' column has 7 or more characters.
    This helps to identify municipalities in the dataset, removing rows that represent Portugal and regions as a whole.
    
    Parameters:
    df (DataFrame): Input DataFrame.
    
    Returns:
    df_municipios (DataFrame): Filtered DataFrame containing only municipalities.
    '''
    df['codigo'] = df['codigo'].astype(str)
    df_municipios = df[df['codigo'].astype(str).apply(lambda x: len(x) == 4)].copy()
    
    translation_table = pd.read_excel("auxiliary/municipality_translation_table.xlsx")

    # Create a mapping dictionary from the translation table
    mapping = dict(zip(translation_table['local'], translation_table['codigo']))

    # Update the 'codigo' column in the DataFrame based on the 'local' column
    df_municipios['codigo'] = df_municipios['local'].map(mapping)

    return df_municipios

def carregar_municipios(df):
    '''
    Filters the DataFrame to include only rows where the 'codigo' column has 7 or more characters.
    This helps to identify municipalities in the dataset, removing rows that represent Portugal and regions as an whole.
    
    Parameters:
    df (DataFrame): Input DataFrame.
    
    Returns:
    df_municipios (DataFrame): Filtered DataFrame containing only municipalities.
    '''
    df['codigo'] = df['codigo'].astype(str)
    df_municipios = df[df['codigo'].astype(str).apply(lambda x: len(x) >= 7)].copy()
    return df_municipios


def aplicar_kmeans(df, n_clusters=3, nome_cluster='cluster'):
    dados_numericos = df.drop(columns=['local', 'codigo'])
    scaler = StandardScaler()
    dados_escalados = scaler.fit_transform(dados_numericos)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(dados_escalados)

    # Ordena os clusters
    centroides = kmeans.cluster_centers_
    ordem = np.argsort(centroides[:, 0])
    novo_mapeamento = {old: new for new, old in enumerate(ordem)}
    labels_ordenados = np.vectorize(novo_mapeamento.get)(labels)

    df[nome_cluster] = labels_ordenados

    coluna = dados_numericos.columns[0]
    df_plot = df.copy()
    df_plot['valor'] = dados_escalados[:, 0]
    df_plot['y'] = 0 

    fig = px.scatter(
        df_plot, 
        x='valor', 
        y='y', 
        color=nome_cluster,
        hover_data=['local', 'codigo', coluna],
        color_continuous_scale='viridis',
    )
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(
        xaxis_title=coluna,
        yaxis=dict(showticklabels=False, showgrid=False),
        height=300
    )
    fig.show()

    print(df[nome_cluster].value_counts().sort_index())
    
    
    return df

def elbow_rule(df, max_k=10):
    X = df.drop(columns=['local', 'codigo'])
    inertias = []
    ks = range(1, max_k + 1)

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker='o')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inércia')
    plt.title('Regra do Cotovelo (Elbow Method)')
    plt.grid(True)
    plt.xticks(ks)
    plt.show()
    
def save_clustered_df_to_csv(df, output_path, file_name):
    
    
    # Select the first two columns and the last column
    columns_to_save = [df.columns[0], df.columns[1], df.columns[-1]]
    df_to_save = df[columns_to_save]
    
    # Construct the full file path
    full_path = os.path.join(output_path, file_name)
    
    # Save the dataframe to a CSV file
    df_to_save.to_csv(full_path, index=False)
    print(f"File saved to {full_path}")