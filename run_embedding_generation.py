# run_embedding_generation.py (CORRIGIDO - Cálculo Pico Geral e Chaves)

import torch
import torch.optim as optim
import os
import time
import random
import numpy as np
import psutil
from functools import partial

# --- IMPORTAR memory_profiler ---
try:
    from memory_profiler import memory_usage

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    print("AVISO: memory_profiler não está instalado (pip install memory_profiler).")
    print("       A medição de pico de memória durante o treino será desativada.")
    MEMORY_PROFILER_AVAILABLE = False
# --- FIM DA IMPORTAÇÃO ---


from src.config import Config
from src.data_loader import get_loader
from src.data_converter import DataConverter
from src.model import VGAE

# Importa as funções de train.py (incluindo format_b)
from src.train import train_model, save_results, save_report, format_b
from src.directory_manager import DirectoryManager

# REMOVIDA: format_bytes agora é importada de train.py
# def format_bytes(b): ...


def main():
    """
    Função principal.
    """
    # --- Configuração Inicial ---
    config = Config()
    device = torch.device(config.DEVICE)

    # --- INICIAR MONITORAMENTO GERAL ---
    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss
    print(f"RAM inicial do processo: {format_b(mem_start)}")

    peak_ram_overall_bytes = mem_start  # Pico geral começa aqui

    if "cuda" in device.type and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        print("VRAM (GPU) Peak Stats zeradas.")

    # ... (seed, prints iniciais) ...
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)
    print("=" * 50)
    print("INICIANDO PROCESSO DE TREINAMENTO E EXTRAÇÃO DE EMBEDDINGS (VGAE)")
    print("=" * 50)
    print(f"Dispositivo de treinamento: {device}")
    print(f"Dataset selecionado: {config.DATASET_NAME}")

    # --- Pipeline de Dados ---
    print("\n[FASE 1/2] Executando pipeline de dados...")
    loader = get_loader(config.DATASET_NAME)
    wsg_obj = loader.load()
    mem_after_load = process.memory_info().rss
    peak_ram_overall_bytes = max(
        peak_ram_overall_bytes, mem_after_load
    )  # Atualiza pico
    print(f"RAM após carregar wsg_obj: {format_b(mem_after_load)}")

    pyg_data = DataConverter.to_pyg_data(wsg_obj, for_embedding_bag=True).to(device)
    mem_after_convert = process.memory_info().rss
    peak_ram_overall_bytes = max(
        peak_ram_overall_bytes, mem_after_convert
    )  # Atualiza pico
    print(
        f"RAM após converter para pyg_data (EmbeddingBag): {format_b(mem_after_convert)}"
    )
    print("Pipeline de dados concluído.")

    directory_manager = DirectoryManager(
        timestamp=config.TIMESTAMP,
        run_folder_name="EMBEDDING_RUNS",  # Corrigindo nome da pasta
    )

    # --- Instanciação do Modelo ---
    print("\n[FASE 3] Construindo o modelo VGAE...")
    model = VGAE(
        num_total_features=pyg_data.num_total_features,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        out_embedding_dim=config.OUT_EMBEDDING_DIM,
    ).to(device)
    mem_after_model = process.memory_info().rss
    peak_ram_overall_bytes = max(
        peak_ram_overall_bytes, mem_after_model
    )  # Atualiza pico
    print(f"RAM após instanciar modelo: {format_b(mem_after_model)}")

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    print("Modelo construído com sucesso.")

    # --- Loop de Treinamento ---
    print("\n[FASE 4] Iniciando treinamento do modelo...")
    start_time = time.perf_counter()

    peak_ram_train_func_mib = 0.0  # Pico durante a função train_model (MiB)

    if MEMORY_PROFILER_AVAILABLE:
        func_to_profile = partial(
            train_model,
            model=model,
            data=pyg_data,
            optimizer=optimizer,
            epochs=config.EPOCHS,
        )
        mem_usage_result, (trained_model, training_history) = memory_usage(
            func_to_profile, max_usage=True, retval=True, interval=0.1
        )
        peak_ram_train_func_mib = mem_usage_result or 0.0  # Garante float
        # Atualiza pico GERAL com o pico do treino (convertido para Bytes)
        peak_ram_overall_bytes = max(
            peak_ram_overall_bytes, int(peak_ram_train_func_mib * 1024 * 1024)
        )
    else:
        print(
            "AVISO: memory_profiler não disponível. Pico de RAM durante treino não medido precisamente."
        )
        trained_model, training_history = train_model(
            model, pyg_data, optimizer, config.EPOCHS
        )
        # Usa memória PÓS treino como estimativa do pico geral
        peak_ram_overall_bytes = max(peak_ram_overall_bytes, process.memory_info().rss)

    end_time = time.perf_counter()
    training_duration = end_time - start_time
    # --- FIM TREINO ---

    # --- Coleta de Métricas Finais ---
    mem_after_train = process.memory_info().rss  # RAM medida *após* o bloco de treino
    # Garante que o pico geral considere o valor final
    peak_ram_overall_bytes = max(peak_ram_overall_bytes, mem_after_train)

    print(f"\nRAM após treinamento (imediatamente após): {format_b(mem_after_train)}")
    print(f"RAM PICO durante a função treino: {format_b(peak_ram_train_func_mib)}")

    peak_vram_bytes = 0
    if "cuda" in device.type and torch.cuda.is_available():
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        print(f"PICO VRAM (GPU) durante treino: {format_b(peak_vram_bytes)}")

    print(f"Treinamento finalizado em {training_duration:.2f} segundos.")

    # --- Inferência e Salvamento ---
    print("\n[FASE FINAL] Gerando e salvando resultados...")
    run_path = directory_manager.get_run_path()

    inference_start_time = time.perf_counter()
    final_embeddings = trained_model.get_embeddings(pyg_data)
    inference_end_time = time.perf_counter()
    inference_duration = inference_end_time - inference_start_time
    print(
        f"Geração de embeddings (inferência) concluída em {inference_duration:.4f} segundos."
    )

    mem_after_inference = process.memory_info().rss  # RAM no final de tudo
    # Garante que o pico geral considere o valor final pós-inferência
    peak_ram_overall_bytes = max(peak_ram_overall_bytes, mem_after_inference)

    # --- DICIONÁRIO DE MÉTRICAS CORRIGIDO ---
    memory_metrics = {
        # Valores em Bytes
        "ram_start_bytes": mem_start,
        "ram_after_load_bytes": mem_after_load,
        "ram_after_convert_bytes": mem_after_convert,
        "ram_after_model_bytes": mem_after_model,
        "ram_after_train_bytes": mem_after_train,  # RAM após o bloco de treino
        "ram_after_inference_bytes": mem_after_inference,  # RAM no final
        "peak_ram_overall_bytes": peak_ram_overall_bytes,  # Pico máximo geral (Bytes)
        "vram_peak_bytes": peak_vram_bytes,
        # Pico da função treino (MiB) - Chave esperada por save_report
        "peak_ram_train_func_MiB": peak_ram_train_func_mib,
        # Versões legíveis - Chaves esperadas por save_report
        "ram_start_readable": format_b(mem_start),
        "ram_after_load_readable": format_b(mem_after_load),
        "ram_after_convert_readable": format_b(mem_after_convert),
        "ram_after_model_readable": format_b(mem_after_model),
        "ram_after_train_readable": format_b(
            mem_after_train
        ),  # Legível RAM após treino
        "ram_after_inference_readable": format_b(
            mem_after_inference
        ),  # Legível RAM final
        "peak_ram_train_func_readable": format_b(
            peak_ram_train_func_mib
        ),  # Pico da func treino formatado
        "peak_ram_overall_readable": format_b(
            peak_ram_overall_bytes
        ),  # Pico GERAL formatado
        "vram_peak_readable": format_b(peak_vram_bytes),
    }

    # Salvar os artefatos
    save_results(trained_model, final_embeddings, wsg_obj, config, save_path=run_path)
    save_report(  # Passa o dict corrigido
        config,
        training_history,
        training_duration,
        inference_duration,
        save_path=run_path,
        memory_metrics=memory_metrics,
    )

    # ... (Resto do script) ...
    final_metrics = training_history[-1] if training_history else {}
    run_metrics = {
        "loss": final_metrics.get("total_loss", 0.0),
        "emb_dim": config.OUT_EMBEDDING_DIM,
    }
    final_path = directory_manager.finalize_run_directory(
        dataset_name=config.DATASET_NAME, metrics=run_metrics
    )
    print("\n" + "=" * 50)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f"Resultados salvos em: '{final_path}'")
    print("=" * 50)


if __name__ == "__main__":
    main()
