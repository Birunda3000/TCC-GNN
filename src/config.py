# src/config.py (versão atualizada)
import torch
import os

class Config:
    """
    Classe para centralizar todas as configurações do projeto.
    """

    # --- Configurações do Dispositivo ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Configurações do Dataset ---
    DATASET_NAME = "Cora"  # Opções: 'Cora', 'MusaeGithub'

    # --- Diretorios Base ---
    ROOT_DIR = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )  # Raiz do projeto gnn_tcc/
    DATA_DIR = os.path.join(ROOT_DIR, "data", "datasets")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "processed")
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Garante que o diretório de saída exista

    # --- Caminhos para os Arquivos Brutos ---
    # Usaremos estes caminhos para carregar os dados manualmente.
    # Cora
    CORA_CONTENT_PATH = "data/datasets/cora/cora.content"
    CORA_CITES_PATH = "data/datasets/cora/cora.cites"
    # Musae-Github
    GITHUB_MUSAE_EDGES_PATH = "data/datasets/musae-github/musae_git_edges.csv"
    GITHUB_MUSAE_FEATURES_PATH = "data/datasets/musae-github/musae_git_features.json"
    GITHUB_MUSAE_TARGET_PATH = "data/datasets/musae-github/musae_git_target.csv"

    # --- Caminhos para os Arquivos Processados ---
    OUTPUT_PATH = "data/output/"
    EXPERIMENTS_LOG_PATH = os.path.join(OUTPUT_PATH, "RUNS/")

    # --- Configurações de Treinamento ---
    SPLIT_RATIO = (0.7, 0.15, 0.15)  # Proporção para (Treino, Validação, Teste)

    # --- Configurações de Visualização ---
    VIS_SAMPLES = 50000
    VIS_OUTPUT_FILENAME = "data/output/graph_visualization.png"

print(f"Configurações carregadas. Usando dispositivo: {Config.DEVICE}")
print(f"Dataset selecionado: {Config.DATASET_NAME}")