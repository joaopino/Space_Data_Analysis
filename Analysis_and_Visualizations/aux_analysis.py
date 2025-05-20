import pandas as pd
import plotly.graph_objects as go
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import json
import geopandas as gpd


def plot_correlacoes_interativas(df):
    corr = df.corr(numeric_only=True)
    np.fill_diagonal(corr.values, np.nan)
    variaveis = corr.columns.tolist()

    # Slider de limiar entre -1.0 e 1.0
    slider = widgets.FloatSlider(
        value=0.7,
        min=-1.0,
        max=1.0,
        step=0.05,
        description='Limiar:',
        continuous_update=False,
        readout_format='.2f',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )

    # Função de atualização
    def atualizar_grafico(limiar):
        if limiar >= 0:
            mask = corr >= limiar
        else:
            mask = corr <= limiar

        filtered_corr = corr.where(mask)

        fig = go.Figure(data=go.Heatmap(
            z=filtered_corr.values,
            x=variaveis,
            y=variaveis,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            zmid=0,
            colorbar=dict(title="Correlação"),
            hovertemplate="Variável X: %{x}<br>Variável Y: %{y}<br>Correlação: %{z:.2f}<extra></extra>"
        ))

        fig.update_layout(
            title=f"Mapa de Correlações | Limiar: {limiar}",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            width=800,
            height=800
        )
        fig.update_yaxes(autorange='reversed')
        fig.show()

    # Exibir
    widgets.interact(atualizar_grafico, limiar=slider)

def plot_correlation_heatmap(df):
    # matriz de correlação
    correlacoes = df.corr(numeric_only=True)
    variaveis = correlacoes.columns.tolist()

    # Heatmap interativo
    fig = go.Figure(data=go.Heatmap(
        z=correlacoes.values,
        x=variaveis,
        y=variaveis,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlação"),
        hovertemplate=(
            "Variável X: %{x}<br>"
            "Variável Y: %{y}<br>"
            "Correlação: %{z:.2f}<extra></extra>"
        )
    ))

    # Ajustar layout
    fig.update_layout(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        title="Mapa de Calor das Correlações (Interativo)",
        width=800,
        height=800
    )

    fig.update_yaxes(autorange='reversed')

    fig.show()

def kmeans(df, features, x_axis, y_axis, n_clusters=3, path = None):
    for col in features + [x_axis, y_axis]:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no DataFrame.")

    dados = df[features].dropna()
    scaler = StandardScaler()
    dados_normalizados = scaler.fit_transform(dados)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(dados_normalizados)

    # DataFrame para visualização
    df_plot = df.loc[dados.index].copy()
    df_plot['cluster'] = clusters

    fig = px.scatter(
        df_plot,
        x=x_axis,
        y=y_axis,
        color=df_plot['cluster'].astype(str),
        title=f"KMeans com {n_clusters} clusters",
        labels={'color': 'Cluster'},
        hover_data=features
    )

    fig.update_layout(width=800, height=600)
    fig.show()

    features.append('local')
    df_resultado = df[features]
    df_resultado.loc[dados.index, 'cluster'] = clusters

    print(df_resultado['cluster'].value_counts().sort_index())

    if path:
        df_resultado.to_csv(path)

    return df_resultado


def top_correlations(df: pd.DataFrame,
                     top_n: int = None,
                     threshold: float = None,
                     drop_odd_rows: bool = True
                    ) -> pd.DataFrame:
    
    # 1) Seleciona só colunas numéricas
    df_num = df.select_dtypes(include='number')

    # 2) Calcula matriz de correlação absoluta
    corr = df_num.corr()

    # 3) Máscara para zero na diagonal
    mask = ~np.eye(corr.shape[0], dtype=bool)

    # 4) Aplica máscara e desempilha
    pairs = (
        corr.where(mask)
            .stack()
            .reset_index()
    )
    pairs.columns = ['var1', 'var2', 'correlation']

    # 5) Filtra por threshold
    if threshold is not None:
        pairs = pairs[pairs['correlation'] >= threshold]

    # 6) Ordena e limita top_n
    pairs = pairs.sort_values('correlation', ascending=False)
    if top_n is not None:
        pairs = pairs.iloc[:top_n]

    # 7) Se solicitado, elimina linhas ímpares (1,3,5...)
    if drop_odd_rows:
        pairs = pairs.iloc[::2].reset_index(drop=True)

    return pairs.reset_index(drop=True)

def drop_low_correlation_vars(df, limiar=0.1):
    df = df.copy()
    cols_para_remover = [col for col in ['local', 'codigo'] if col in df.columns]
    df = df.drop(columns=cols_para_remover)
    corr_mat = df.corr().abs()

    # para cada coluna, ignora auto-correlação (1.0) e acha o máximo com as outras
    max_corr = corr_mat.where(~np.eye(len(corr_mat), dtype=bool)).max()

    # seleciona colunas cujo máximo é >= limiar
    manter = max_corr[max_corr >= limiar].index
    remover = max_corr[max_corr < limiar].index

    df_filtrado = df[manter]
    return df_filtrado,list(remover)


def clustering(df, variaveis,nome, n_clusters=None, max_k=10, random_state=42,path = None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    # 1. Garante que a coluna 'local' existe
    if 'local' not in df.columns:
        raise ValueError("A coluna 'local' não existe no DataFrame.")

    # 2. Dropar apenas linhas com NaN nas variáveis
    df_valid = df.dropna(subset=variaveis).copy()

    # 3. Normalizar apenas as variáveis
    X = df_valid[variaveis]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 5. Gráfico do cotovelo
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X_pca)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Inércia")
    plt.title("Gráfico do Cotovelo")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if n_clusters is None:
        print("Escolhe o número ideal de clusters com base no gráfico do cotovelo e chama novamente com `n_clusters=`.")
        return None

    # 6. Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(X_pca)

    # 7. Resultado final com 'local'
    df_resultado = df_valid[['local'] + variaveis].copy()
    df_resultado['PCA1'] = X_pca[:, 0]
    df_resultado['PCA2'] = X_pca[:, 1]
    df_resultado[nome] = clusters

    # 8. Visualização
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_resultado,
        x='PCA1',
        y='PCA2',
        hue=nome,
        palette='Set2',
        s=100
    )
    plt.title(f'Clusters de Vulnerabilidade Social (k={n_clusters})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if path:
        df_resultado.to_csv(path)

    return df_resultado

def analisar_variaveis_por_cluster(df, cluster_col, variaveis, incluir_coluna_extra=None):
    """
    Exibe a contagem absoluta de cada variável categórica por cluster.

    Parâmetros:
    - df (pd.DataFrame): DataFrame com os dados.
    - cluster_col (str): Nome da coluna de cluster.
    - variaveis (list): Lista de nomes das variáveis categóricas a analisar.
    - incluir_coluna_extra (str ou None): Nome de uma coluna adicional para exibir (ex: 'local').

    Retorna:
    - None (exibe as tabelas diretamente).
    """
    if incluir_coluna_extra:
        display(df[[incluir_coluna_extra, cluster_col]].head())

    for var in variaveis:
        print(f"Contagem absoluta de '{var}' dentro de cada '{cluster_col}':")
        tabela = pd.crosstab(
            df[cluster_col],
            df[var],
            margins=True,
            normalize=False
        )
        display(tabela)
