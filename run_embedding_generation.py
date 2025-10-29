# run_embedding_generation.py (Corrigido para capturar o PICO de RAM)

import torch
import torch.optim as optim
import os
import time
import random
import numpy as np
import psutil  

from src.config import Config
from src.data_loader import get_loader
from src.data_converter import DataConverter
from src.model import VGAE
from src.train import train_model, save_results, save_report # Importa a função atualizada
from src.directory_manager import DirectoryManager


def format_bytes(b):
    """Converte bytes para um formato legível (MB ou GB)."""
    if b < 1024**3:
        return f"{b / 1024**2:.2f} MB"
    return f"{b / 1024**3:.2f} GB"


def main():
    """
    Função principal que orquestra todo o processo de treinamento e
    extração de embeddings para o modelo VGAE.
    """
    # --- 1. Configuração Inicial ---
    config = Config()
    device = torch.device(config.DEVICE)

    # --- 2. INICIAR MONITORAMENTO DE MEMÓRIA ---
    process = psutil.Process(os.getpid()) # <--- Objeto 'process' é criado
    mem_start = process.memory_info().rss  
    print(f"RAM inicial do processo: {format_bytes(mem_start)}")
    
    if "cuda" in device.type and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        print("VRAM (GPU) Peak Stats zeradas.")
    # --- FIM DO SETUP DE MONITORAMENTO ---

    # Aplica a semente de aleatoriedade
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    print("=" * 50)
    print("INICIANDO PROCESSO DE TREINAMENTO E EXTRAÇÃO DE EMBEDDINGS")
    print("=" * 50)
    print(f"Dispositivo de treinamento: {device}")
    print(f"Dataset selecionado: {config.DATASET_NAME}")

    # --- 3. Pipeline de Dados (Loader -> WSG -> Converter -> PyG) ---
    print("\n[FASE 1/2] Executando pipeline de dados...")
    loader = get_loader(config.DATASET_NAME)
    wsg_obj = loader.load()
    mem_after_load = process.memory_info().rss
    print(f"RAM após carregar wsg_obj: {format_bytes(mem_after_load)}")

    pyg_data = DataConverter.to_pyg_data(wsg_obj, for_embedding_bag=True).to(device)
    mem_after_convert = process.memory_info().rss
    print(f"RAM após converter para pyg_data (EmbeddingBag): {format_bytes(mem_after_convert)}")
    print("Pipeline de dados concluído. Dados prontos para o modelo.")


    directory_manager = DirectoryManager(
        timestamp=config.TIMESTAMP,
        run_folder_name="EMBEDDING_RUNS"
    )


    # --- 4. Instanciação do Modelo e Otimizador ---
    print("\n[FASE 3] Construindo o modelo VGAE...")
    model = VGAE(
        num_total_features=pyg_data.num_total_features,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        out_embedding_dim=config.OUT_EMBEDDING_DIM,
    ).to(device)
    mem_after_model = process.memory_info().rss
    print(f"RAM após instanciar modelo: {format_bytes(mem_after_model)}")
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    print("Modelo construído com sucesso.")
    print(model)

    # --- 5. Loop de Treinamento ---
    print("\n[FASE 4] Iniciando treinamento do modelo...")
    start_time = time.time()
    
    # --- 1. CHAMADA DE FUNÇÃO ATUALIZADA ---
    # Passamos o 'process' e recebemos o 'peak_ram_during_train' de volta
    trained_model, training_history, peak_ram_during_train = train_model(
        model, pyg_data, optimizer, config.EPOCHS, process
    )
    # --- FIM DA ATUALIZAÇÃO ---
    
    end_time = time.time()
    training_duration = end_time - start_time

    # --- 6. COLETAR MÉTRICAS DE PICO ---
    mem_after_train = process.memory_info().rss
    print(f"RAM após treinamento (Final): {format_bytes(mem_after_train)}")
    print(f"RAM PICO durante treinamento: {format_bytes(peak_ram_during_train)}") # Log
    
    peak_vram_bytes = 0
    if "cuda" in device.type and torch.cuda.is_available():
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        print(f"PICO VRAM (GPU) durante treino: {format_bytes(peak_vram_bytes)}")
    
    print(f"Treinamento finalizado em {training_duration:.2f} segundos.")

    # --- 7. Extração e Salvamento dos Resultados ---
    print("\n[FASE FINAL] Gerando e salvando resultados...")
    run_path = directory_manager.get_run_path()

    inference_start_time = time.time()
    final_embeddings = trained_model.get_embeddings(pyg_data)
    inference_end_time = time.time()
    inference_duration = inference_end_time - inference_start_time
    print(f"Geração de embeddings (inferência) concluída em {inference_duration:.4f} segundos.")

    # --- 2. DICIONÁRIO DE MÉTRICAS ATUALIZADO ---
    memory_metrics = {
        "ram_start_bytes": mem_start,
        "ram_after_load_bytes": mem_after_load,
        "ram_after_convert_bytes": mem_after_convert,
        "ram_after_model_bytes": mem_after_model,
        "ram_after_train_bytes": mem_after_train,
        "ram_peak_during_train_bytes": peak_ram_during_train, # <-- ADICIONADO
        "ram_peak_during_train_readable": format_bytes(peak_ram_during_train), # <-- ADICIONADO
        "vram_peak_bytes": peak_vram_bytes,
        "vram_peak_readable": format_bytes(peak_vram_bytes),
    }

    # Salvar os artefatos da execução
    save_results(trained_model, final_embeddings, wsg_obj, config, save_path=run_path)
    save_report(
        config,
        training_history,
        training_duration,
        inference_duration,
        save_path=run_path,
        memory_metrics=memory_metrics, # Passa o dict atualizado
    )

    # Prepara as métricas para o nome do diretório final
    final_metrics = training_history[-1]
    run_metrics = {
        "loss": final_metrics.get("total_loss", 0.0),
        "emb_dim": config.OUT_EMBEDDING_DIM,
    }

    # Finaliza o diretório de execução
    final_path = directory_manager.finalize_run_directory(
        dataset_name=config.DATASET_NAME, metrics=run_metrics
    )

    print("\n" + "=" * 50)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f"Resultados salvos em: '{final_path}'")
    print("=" * 50)


if __name__ == "__main__":
    main()