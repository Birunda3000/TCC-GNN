import torch
import torch.optim as optim
import os

from src.config import Config
from src.data_loader import get_loader
from src.data_converter import DataConverter
from src.model import VGAE
from src.train import train_model, save_results


def main():
    """
    Função principal que orquestra todo o processo de treinamento e
    extração de embeddings para o modelo VGAE.
    """
    # --- 1. Configuração Inicial ---
    config = Config()
    device = torch.device(config.DEVICE)

    print("=" * 50)
    print("INICIANDO PROCESSO DE TREINAMENTO E EXTRAÇÃO DE EMBEDDINGS")
    print("=" * 50)
    print(f"Dispositivo de treinamento: {device}")
    print(f"Dataset selecionado: {config.DATASET_NAME}")

    # --- 2. Pipeline de Dados (Loader -> WSG -> Converter -> PyG) ---
    print("\n[FASE 1/2] Executando pipeline de dados...")
    loader = get_loader(config.DATASET_NAME)
    wsg_obj = loader.load()
    pyg_data = DataConverter.to_pyg_data(wsg_obj).to(device)
    print("Pipeline de dados concluído. Dados prontos para o modelo.")

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
    trained_model = train_model(model, pyg_data, optimizer, config.EPOCHS)
    print("Treinamento finalizado.")

    # --- 5. Extração e Salvamento dos Resultados ---
    print("\n[FASE FINAL] Gerando e salvando resultados...")
    save_results(trained_model, pyg_data, wsg_obj, config)

    print("\n" + "=" * 50)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print("=" * 50)


if __name__ == "__main__":
    main()
