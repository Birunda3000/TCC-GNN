# scripts/visualize_final_results_2D.py

import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# A projeção 3D não é mais necessária
# from mpl_toolkits.mplot3d import Axes3D

# --- 1. CONFIGURAÇÕES ---
# O único arquivo de entrada que precisamos do nosso treinamento
EMBEDDINGS_FILE = 'data/output/embeddings_output_20250917_174303.json'

# O arquivo bruto original que contém os labels para colorirmos o gráfico
RAW_TARGET_PATH = 'data/datasets/musae-github/musae_git_target.csv'

# --- 2. EXECUÇÃO PRINCIPAL ---

def main():
    """
    Script autocontido para carregar embeddings pré-treinados e visualizá-los.
    Lê apenas o arquivo de resultados e o arquivo de labels original.
    """
    print("Iniciando script de visualização final (2D)...")

    # --- Carregar os Dados Necessários ---
    try:
        print(f"Carregando embeddings de: {EMBEDDINGS_FILE}")
        with open(EMBEDDINGS_FILE, 'r') as f:
            results = json.load(f)

        print(f"Carregando labels de: {RAW_TARGET_PATH}")
        target_df = pd.read_csv(RAW_TARGET_PATH)

    except FileNotFoundError as e:
        print(f"\nERRO: Arquivo não encontrado: {e.filename}")
        print("Certifique-se de que os caminhos no topo do script estão corretos.")
        return

    # --- Preparar os Dados para Análise ---
    metadata = results['metadata']
    num_nodes = metadata['num_nodes']
    embedding_dim = metadata['embedding_dim']

    embeddings = np.zeros((num_nodes, embedding_dim))
    for i in range(num_nodes):
        node_id_str = str(i)
        if node_id_str in results['nodes']:
            embeddings[i] = results['nodes'][node_id_str]['embedding']

    target_df = target_df.sort_values(by='id').reset_index(drop=True)
    labels = target_df['ml_target'].to_numpy()

    if len(labels) != num_nodes:
        print(f"AVISO: Inconsistência de tamanho. {num_nodes} nós nos embeddings, mas {len(labels)} labels encontrados.")
        min_size = min(num_nodes, len(labels))
        embeddings = embeddings[:min_size]
        labels = labels[:min_size]
        
    print(f"Dados prontos. Shape dos embeddings: {embeddings.shape}")

    # --- Aplicar PCA ---
    # MUDANÇA 1: Reduzir para 2 componentes em vez de 3
    print("Aplicando PCA para reduzir para 2 dimensões...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    variance_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"Variância explicada pelas 2 componentes: {variance_explained:.2f}%")

    # --- Visualizar os Resultados ---
    # MUDANÇA 2: Usar o código de plotagem 2D do Matplotlib
    print("Gerando visualização 2D...")
    fig, ax = plt.subplots(figsize=(12, 8)) # Cria uma figura e um eixo 2D

    indices_web = np.where(labels == 0)[0]
    indices_ml = np.where(labels == 1)[0]

    # Plota os desenvolvedores Web (azul) em 2D
    ax.scatter(embeddings_2d[indices_web, 0], embeddings_2d[indices_web, 1],
               c='blue', label=f'Web Developers ({len(indices_web)})', alpha=0.6, s=15)

    # Plota os desenvolvedores de ML (laranja) em 2D
    ax.scatter(embeddings_2d[indices_ml, 0], embeddings_2d[indices_ml, 1],
               c='orange', label=f'ML Developers ({len(indices_ml)})', alpha=0.6, s=15)

    ax.set_title(f'Visualização 2D dos Embeddings (PCA) - {metadata["dataset_name"]}\nVariância Explicada: {variance_explained:.2f}%')
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # MUDANÇA 3: Atualizar o nome do arquivo de saída
    output_filename = 'github_embeddings_visualization_final_2d.png'
    plt.savefig(output_filename)
    print(f"\nGráfico salvo como '{output_filename}'. Exibindo agora...")
    plt.show()


if __name__ == '__main__':
    main()