# src/config.py (versão atualizada)
import os
from datetime import datetime
from zoneinfo import ZoneInfo


class Config:
    """
    Classe centralizada para todas as configurações do projeto.
    """

    # --- Timestamp da Execução ---
    # Gera um timestamp único no momento da inicialização para identificar a execução.
    # Usa o fuso horário de São Paulo para consistência.
    TIMESTAMP = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime(
        "%d-%m-%Y_%H-%M-%S"
    )

    # --- Configurações do Ambiente ---
    DEVICE = "cuda" if os.environ.get("NVIDIA_VISIBLE_DEVICES") else "cpu"
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")

    # --- Caminhos de Saída ---
    OUTPUT_PATH = os.path.join(DATA_DIR, "output")

    # --- Configurações do Dataset ---
    # Opções: "cora", "musae-github", "musae-facebook"
    DATASET_NAME = "musae-github"

    # --- Caminhos para os arquivos do Musae-Github ---
    GITHUB_MUSAE_EDGES_PATH = os.path.join(
        DATA_DIR, "datasets", "musae-github", "musae_git_edges.csv"
    )
    GITHUB_MUSAE_TARGET_PATH = os.path.join(
        DATA_DIR, "datasets", "musae-github", "musae_git_target.csv"
    )
    GITHUB_MUSAE_FEATURES_PATH = os.path.join(
        DATA_DIR, "datasets", "musae-github", "musae_git_features.json"
    )

    # --- Hiperparâmetros do Modelo VGAE ---
    EMBEDDING_DIM = 128  # Dimensão do embedding das features de entrada
    HIDDEN_DIM = 64  # Dimensão da camada GCN oculta
    OUT_EMBEDDING_DIM = 8  # Dimensão do embedding final do nó

    # --- Configurações de Treinamento ---
    EPOCHS = 100
    LEARNING_RATE = 0.01

    # --- Configurações de Visualização ---
    VIS_SAMPLES = 1500  # Número máximo de nós para incluir na visualização


print(f"Configurações carregadas. Usando dispositivo: {Config.DEVICE}")
print(f"Dataset selecionado: {Config.DATASET_NAME}")
