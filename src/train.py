# src/train.py (Corrigido para capturar o PICO de RAM)

import torch
import torch.optim as optim
import os
import json
from datetime import datetime
import pytz
import time
from typing import Tuple, Dict, List, Optional, Any
import psutil  # <--- 1. IMPORTADO

# --- Importa nossos módulos customizados ---
from src.config import Config
from src.data_loader import get_loader
from src.data_converter import DataConverter
from src.model import VGAE
from src.data_format_definition import WSG, Metadata, GraphStructure, NodeFeaturesEntry
from torch_geometric.data import Data


def train_model(
    model: VGAE,
    data: Data,
    optimizer: optim.Optimizer,
    epochs: int,
    process: psutil.Process,  # <--- 2. NOVO ARGUMENTO
) -> Tuple[VGAE, List[Dict[str, float]], int]:  # <--- 3. RETORNO ATUALIZADO
    """
    Executa o loop de treinamento para o modelo VGAE.
    Agora também monitora o pico de RAM durante o treinamento.

    Args:
        model (VGAE): A instância do modelo a ser treinado.
        data (Data): Os dados do grafo no formato PyG.
        optimizer (optim.Optimizer): O otimizador.
        epochs (int): O número de épocas para treinar.
        process (psutil.Process): O objeto do processo atual para monitorar a RAM.

    Returns:
        Tuple[VGAE, List[Dict[str, float]], int]: O modelo treinado, um histórico
                                                das métricas de loss e o pico de
                                                RAM (em bytes) observado.
    """
    training_history = []
    # 4. Inicializa o pico de RAM com o valor atual
    peak_ram_during_train = process.memory_info().rss
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encode(data)
        recon_loss = model.reconstruction_loss(z, data.edge_index)
        kl_loss = (1 / data.num_nodes) * model.kl_loss()
        total_loss = recon_loss + kl_loss
        
        total_loss.backward()
        optimizer.step()

        # --- 5. LÓGICA DE MONITORAMENTO DE PICO DE RAM ---
        # A cada época, verifica o uso de RAM após o backpropagation
        current_ram = process.memory_info().rss
        if current_ram > peak_ram_during_train:
            peak_ram_during_train = current_ram
        # --- FIM DA LÓGICA ---

        epoch_metrics = {
            "epoch": epoch,
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
        }
        training_history.append(epoch_metrics)

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch:03d} | Loss Total: {total_loss:.4f} | "
                f"Reconstrução: {recon_loss:.4f} | KL: {kl_loss:.4f} | "
                f"RAM Pico: {(peak_ram_during_train / 1024**2):.2f} MB" # Feedback
            )

    print(f"Métricas finais do treinamento: {training_history[-1]}")
    # 6. Retorna o pico de RAM
    return model, training_history, peak_ram_during_train


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
    (Atualizado para exibir a nova métrica de PICO DE RAM)
    """
    final_metrics = training_history[-1]
    report_path = os.path.join(save_path, "run_summary.txt")

    # --- 7. SEÇÃO DE MEMÓRIA ATUALIZADA ---
    memory_report = "MÉTRICAS DE MEMÓRIA\n-----------------\n"
    if memory_metrics:
        # Função interna para formatar bytes
        def format_b(b):
            if b < 1024**3:
                return f"{b / 1024**2:.2f} MB"
            return f"{b / 1024**3:.2f} GB"

        memory_report += f"- RAM Inicial: {format_b(memory_metrics.get('ram_start_bytes', 0))}\n"
        memory_report += f"- RAM Após Carregar WSG: {format_b(memory_metrics.get('ram_after_load_bytes', 0))}\n"
        memory_report += f"- RAM Após Converter (EmbBag): {format_b(memory_metrics.get('ram_after_convert_bytes', 0))}\n"
        memory_report += f"- RAM Pós-Treino (Final): {format_b(memory_metrics.get('ram_after_train_bytes', 0))}\n"
        memory_report += f"- PICO DE RAM (Durante Treino): {memory_metrics.get('ram_peak_during_train_readable', 'N/A')}\n" # <-- LINHA ATUALIZADA
        memory_report += f"- PICO de VRAM (GPU): {memory_metrics.get('vram_peak_readable', 'N/A')}\n"
    else:
        memory_report += "- Monitoramento de memória não foi fornecido.\n"
    # --- FIM DA SEÇÃO ---

    content = f"""
=================================================
          RESUMO DA EXECUÇÃO DO MODELO
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
- Loss Total Final: {final_metrics['total_loss']:.6f}
- Loss de Reconstrução Final: {final_metrics['recon_loss']:.6f}
- Loss KL Final: {final_metrics['kl_loss']:.6f}

{memory_report}
=================================================
"""
    with open(report_path, "w") as f:
        f.write(content)

    print(f"Relatório de execução salvo em: '{report_path}'")


def save_results(
    model: VGAE,
    final_embeddings: torch.Tensor,
    wsg_obj: WSG,
    config: Config,
    save_path: str,
):
    """
    Gera os embeddings finais e os salva em um novo arquivo no formato WSG.
    (Sem alterações)
    """
    model.eval()
    tz_info = pytz.timezone("America/Sao_Paulo")

    # --- 1. Salvar o estado do modelo treinado ---
    model_path = os.path.join(save_path, "vgae_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Modelo treinado salvo em: '{model_path}'")

    # --- 2. Usar os embeddings finais já gerados ---
    final_embeddings = final_embeddings.cpu()

    # --- 3. Construir o novo objeto WSG com os embeddings como features ---
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

    # --- 4. Salvar o novo arquivo .wsg.json ---
    output_filename = f"{config.DATASET_NAME}_embeddings.wsg.json"
    output_path = os.path.join(save_path, output_filename)
    with open(output_path, "w") as f:
        f.write(output_wsg.model_dump_json(indent=2))
    print(f"Arquivo de embeddings no formato WSG salvo em: '{output_path}'")


# =========================================================================
# A FUNÇÃO MAIN() ABAIXO ESTÁ OBSOLETA E NÃO É USADA.
# =========================================================================
def main():
    pass # Ignorada

if __name__ == "__main__":
    main()