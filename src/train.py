# src/train.py (CORRIGIDO save_report)

import torch
import torch.optim as optim
import os
import json
from datetime import datetime
import pytz # Ou from zoneinfo import ZoneInfo
import time
from typing import Tuple, Dict, List, Optional, Any

# --- Importa nossos módulos customizados ---
from src.config import Config
from src.model import VGAE
from src.data_format_definition import WSG, Metadata, GraphStructure, NodeFeaturesEntry
from torch_geometric.data import Data
from tqdm import tqdm # Import tqdm if used inside train_model


# --- FUNÇÃO HELPER format_b ATUALIZADA ---
def format_b(b):
    """Converte bytes ou MiB para um formato legível (MB ou GB). Mais robusta."""
    if b is None or b == 0 or b == 0.0:
        # Retorna 0.00 MB para valores nulos ou zero
        return "0.00 MB"

    try:
        if isinstance(b, float): # memory_profiler retorna MiB (float)
            # Converte MiB para Bytes
            b_bytes = int(b * 1024 * 1024)
        elif isinstance(b, int):
            # Já está em bytes
            b_bytes = b
        else:
            # Tipo inesperado
            return "N/A (type)"

        # Formata Bytes para MB/GB
        if b_bytes < 1024**3:
            return f"{b_bytes / 1024**2:.2f} MB"
        else:
            return f"{b_bytes / 1024**3:.2f} GB"
    except Exception:
        # Captura qualquer erro de conversão/formatação
        return "N/A (error)"
# --- FIM FUNÇÃO HELPER ---


# train_model (versão simplificada sem psutil interno)
def train_model(
    model: VGAE,
    data: Data,
    optimizer: optim.Optimizer,
    epochs: int,
) -> Tuple[VGAE, List[Dict[str, float]]]:
    """
    Executa o loop de treinamento para o modelo VGAE.
    """
    training_history = []
    pbar = tqdm(range(1, epochs + 1), desc="Treinando VGAE")

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        z = model.encode(data)
        recon_loss = model.reconstruction_loss(z, data.edge_index)
        kl_loss = (1 / data.num_nodes) * model.kl_loss()
        total_loss = recon_loss + kl_loss
        total_loss.backward()
        optimizer.step()

        epoch_metrics = {
            "epoch": epoch,
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
        }
        training_history.append(epoch_metrics)

        pbar.set_postfix({
            "Loss": f"{total_loss:.4f}",
            "Recon": f"{recon_loss:.4f}",
            "KL": f"{kl_loss:.4f}"
        })

    if training_history:
        print(f"Métricas finais do treinamento: {training_history[-1]}")
    else:
        print("Treinamento concluído sem histórico.")
    return model, training_history


# save_report (CORRIGIDO para usar as chaves corretas)
def save_report(
    config: Config,
    training_history: List[Dict[str, float]],
    training_duration: float,
    inference_duration: float,
    save_path: str,
    memory_metrics: Optional[Dict[str, Any]] = None,
):
    """
    Salva um relatório de texto com o resumo da execução.
    (CORRIGIDO para usar as chaves de memória corretas)
    """
    final_metrics = training_history[-1] if training_history else {}
    report_path = os.path.join(save_path, "run_summary.txt")

    # --- SEÇÃO DE MEMÓRIA CORRIGIDA ---
    memory_report = "MÉTRICAS DE MEMÓRIA\n-----------------\n"
    if memory_metrics:
        # Imprime o dicionário recebido para debug (OPCIONAL)
        # print(f"DEBUG: memory_metrics recebido em save_report: {memory_metrics}")

        memory_report += f"- RAM Inicial: {format_b(memory_metrics.get('ram_start_bytes'))}\n" # Chave original estava correta
        memory_report += f"- RAM Após Carregar WSG: {format_b(memory_metrics.get('ram_after_load_bytes'))}\n" # Chave correta
        memory_report += f"- RAM Após Converter (EmbBag): {format_b(memory_metrics.get('ram_after_convert_bytes'))}\n" # Chave correta
        memory_report += f"- RAM Após Instanciar Modelo: {format_b(memory_metrics.get('ram_after_model_bytes'))}\n" # Chave correta
        memory_report += f"- PICO DE RAM (Durante Treino - func): {format_b(memory_metrics.get('peak_ram_train_func_MiB'))}\n" # Chave correta (espera MiB)
        memory_report += f"- RAM Final (Pós-Inferência): {format_b(memory_metrics.get('ram_after_inference_bytes'))}\n" # <-- CHAVE CORRIGIDA
        memory_report += f"- PICO DE RAM (Geral - Dados/Modelo/Treino): {format_b(memory_metrics.get('peak_ram_overall_bytes'))}\n" # Chave correta
        memory_report += f"- PICO de VRAM (GPU): {format_b(memory_metrics.get('vram_peak_bytes'))}\n" # Chave correta
    else:
        memory_report += "- Monitoramento de memória não foi fornecido.\n"
    # --- FIM DA SEÇÃO ---

    content = f"""
=================================================
          RESUMO DA EXECUÇÃO DO MODELO (VGAE)
=================================================

INFORMAÇÕES GERAIS
------------------
- Dataset: {config.DATASET_NAME}
- Modelo: VGAE
- Timestamp: {config.TIMESTAMP}

CONFIGURAÇÃO DE REPRODUTIBILIDADE
---------------------------------
- Semente Aleatória (seed): {config.RANDOM_SEED}

HIPERPARÂMETROS
---------------
- Épocas de Treinamento: {config.EPOCHS}
- Taxa de Aprendizagem (LR): {config.LEARNING_RATE}
- Dimensão do Embedding de Saída: {config.OUT_EMBEDDING_DIM}
- Dimensão da Camada Oculta (GCN): {config.HIDDEN_DIM}
- Dimensão do Embedding de Features: {config.EMBEDDING_DIM}

RESULTADOS FINAIS
-----------------
- Tempo Total de Treinamento: {training_duration:.2f} segundos
- Tempo de Inferência (gerar embeddings): {inference_duration:.4f} segundos
- Loss Total Final: {final_metrics.get('total_loss', 'N/A'):.6f}
- Loss Reconstrução Final: {final_metrics.get('recon_loss', 'N/A'):.6f}
- Loss KL Final: {final_metrics.get('kl_loss', 'N/A'):.6f}

{memory_report}
=================================================
"""
    with open(report_path, "w") as f:
        f.write(content)

    print(f"Relatório de execução salvo em: '{report_path}'")


# ... (save_results permanece o mesmo) ...
def save_results(
    model: VGAE,
    final_embeddings: torch.Tensor,
    wsg_obj: WSG,
    config: Config,
    save_path: str,
):
    """
    Gera os embeddings finais e os salva em um novo arquivo no formato WSG.
    """
    model.eval()
    try:
        # Use ZoneInfo se Python >= 3.9
        from zoneinfo import ZoneInfo
        tz_info = ZoneInfo("America/Sao_Paulo")
    except ImportError:
        # Fallback para pytz
        import pytz
        tz_info = pytz.timezone("America/Sao_Paulo")

    model_path = os.path.join(save_path, "vgae_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Modelo treinado salvo em: '{model_path}'")

    final_embeddings = final_embeddings.cpu()

    output_metadata = Metadata(
        dataset_name=f"{wsg_obj.metadata.dataset_name}-Embeddings",
        feature_type="dense_continuous",
        num_nodes=wsg_obj.metadata.num_nodes,
        num_edges=wsg_obj.metadata.num_edges,
        num_total_features=config.OUT_EMBEDDING_DIM,
        processed_at=datetime.now(tz_info).isoformat(),
        directed=wsg_obj.metadata.directed,
    )
    output_graph_structure = wsg_obj.graph_structure
    output_node_features = {}
    embedding_indices = list(range(config.OUT_EMBEDDING_DIM))
    for i in range(wsg_obj.metadata.num_nodes):
        node_embedding = final_embeddings[i].tolist()
        output_node_features[str(i)] = NodeFeaturesEntry(
            indices=embedding_indices,
            weights=node_embedding,
        )
    output_wsg = WSG(
        metadata=output_metadata,
        graph_structure=output_graph_structure,
        node_features=output_node_features,
    )

    output_filename = f"{config.DATASET_NAME}_embeddings.wsg.json"
    output_path = os.path.join(save_path, output_filename)
    with open(output_path, "w") as f:
        f.write(output_wsg.model_dump_json(indent=2))
    print(f"Arquivo de embeddings no formato WSG salvo em: '{output_path}'")

# Função main obsoleta omitida para clareza