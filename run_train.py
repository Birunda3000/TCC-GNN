import torch
import torch.optim as optim
import os
import time
import random
import numpy as np

from src.config import Config
from src.data_loader import get_loader
from src.data_converter import DataConverter
from src.model import VGAE
from src.train import train_model, save_results, save_report
from src.directory_manager import DirectoryManager


def main():
    """
    Função principal que orquestra todo o processo de treinamento e
    extração de embeddings para o modelo VGAE.
    """
    # --- 1. Configuração Inicial ---
    config = Config()
    device = torch.device(config.DEVICE)

    # Aplica a semente de aleatoriedade para reprodutibilidade
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    print("=" * 50)
    print("INICIANDO PROCESSO DE TREINAMENTO E EXTRAÇÃO DE EMBEDDINGS")
    print("=" * 50)
    print(f"Dispositivo de treinamento: {device}")
    print(f"Dataset selecionado: {config.DATASET_NAME}")

    # --- 2. Pipeline de Dados (Loader -> WSG -> Converter -> PyG) e salvamento ---
    print("\n[FASE 1/2] Executando pipeline de dados...")
    loader = get_loader(config.DATASET_NAME)
    wsg_obj = loader.load()
    pyg_data = DataConverter.to_pyg_data(wsg_obj).to(device)
    print("Pipeline de dados concluído. Dados prontos para o modelo.")


    directory_manager = DirectoryManager(timestamp=config.TIMESTAMP, base_path=config.OUTPUT_PATH, dataset_name=config.DATASET_NAME)


    # --- 3. Instanciação do Modelo e Otimizador ---
    print("\n[FASE 3] Construindo o modelo VGAE...")
    model = VGAE(
        num_total_features=pyg_data.num_total_features,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        out_embedding_dim=config.OUT_EMBEDDING_DIM,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    print("Modelo construído com sucesso.")
    print(model)

    # --- 4. Loop de Treinamento ---
    print("\n[FASE 4] Iniciando treinamento do modelo...")
    start_time = time.time()
    trained_model, training_history = train_model(model, pyg_data, optimizer, config.EPOCHS)
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Treinamento finalizado em {training_duration:.2f} segundos.")

    # --- 5. Extração e Salvamento dos Resultados ---
    print("\n[FASE FINAL] Gerando e salvando resultados...")
    run_path = directory_manager.get_run_path()
    save_results(trained_model, pyg_data, wsg_obj, config, save_path=run_path)
    save_report(config, training_history, training_duration, save_path=run_path)

    # Prepara as métricas para o nome do diretório final
    final_metrics = training_history[-1]
    run_metrics = {
        "loss": final_metrics.get("total_loss", 0.0),
        "emb_dim": config.OUT_EMBEDDING_DIM,
    }

    # Finaliza o diretório de execução com o nome contendo as métricas
    final_path = directory_manager.finalize_run_directory(
        dataset_name=config.DATASET_NAME, metrics=run_metrics
    )

    print("\n" + "=" * 50)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f"Resultados salvos em: '{final_path}'")
    print("=" * 50)


if __name__ == "__main__":
    main()
