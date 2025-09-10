# src/config.py (versão atualizada)

import torch


class Config:
    """
    Classe para centralizar todas as configurações do projeto.
    """

    # --- Configurações do Dispositivo ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Configurações do Dataset ---
    DATASET_NAME = "Cora"  # Opções: 'Cora', 'MusaeGithub'

    # --- Caminhos para os Arquivos Brutos ---
    # Usaremos estes caminhos para carregar os dados manualmente.
    CORA_CONTENT_PATH = "data/datasets/cora/cora.content"
    CORA_CITES_PATH = "data/datasets/cora/cora.cites"

    GITHUB_MUSAE_EDGES_PATH = "data/datasets/musae-github/musae_git_edges.csv"
    GITHUB_MUSAE_FEATURES_PATH = "data/datasets/musae-github/musae_git_features.json"
    GITHUB_MUSAE_TARGET_PATH = "data/datasets/musae-github/musae_git_target.csv"

    OUTPUT_PATH = "data/output/"

    # --- Configurações de Treinamento ---
    SPLIT_RATIO = (0.7, 0.15, 0.15)  # Proporção para (Treino, Validação, Teste)

    # --- Configurações de Visualização ---
    VIS_SAMPLES = 50000
    VIS_OUTPUT_FILENAME = "data/output/graph_visualization.png"


print(f"Configurações carregadas. Usando dispositivo: {Config.DEVICE}")
print(f"Dataset selecionado: {Config.DATASET_NAME}")
