# train.py

import torch
import torch.optim as optim
import os
import json
from datetime import datetime
import pytz
import time
from typing import Tuple, Dict, List

# --- Importa nossos módulos customizados ---
from src.config import Config
from src.data_loader import get_loader
from src.data_converter import DataConverter
from src.model import VGAE
from src.data_format_definition import WSG, Metadata, GraphStructure, NodeFeaturesEntry
from torch_geometric.data import Data


def train_model(
    model: VGAE, data: Data, optimizer: optim.Optimizer, epochs: int
) -> Tuple[VGAE, List[Dict[str, float]]]:
    """
    Executa o loop de treinamento para o modelo VGAE.

    Args:
        model (VGAE): A instância do modelo a ser treinado.
        data (Data): Os dados do grafo no formato PyG.
        optimizer (optim.Optimizer): O otimizador.
        epochs (int): O número de épocas para treinar.

    Returns:
        Tuple[VGAE, List[Dict[str, float]]]: O modelo treinado e um histórico
                                             das métricas de loss por época.
    """
    training_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encode(data)
        recon_loss = model.reconstruction_loss(z, data.edge_index)

        # A loss KL é frequentemente escalonada pelo número de nós para balancear
        # com a loss de reconstrução, que cresce com o número de arestas.
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

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch:03d} | Loss Total: {total_loss:.4f} | "
                f"Reconstrução: {recon_loss:.4f} | KL: {kl_loss:.4f}"
            )

    print(f"Métricas finais do treinamento: {training_history[-1]}")
    return model, training_history


def save_report(
    config: Config,
    training_history: List[Dict[str, float]],
    training_duration: float,
    inference_duration: float,
    save_path: str,
):
    """
    Salva um relatório de texto com o resumo da execução.

    Args:
        config (Config): O objeto de configuração da execução.
        training_history (List[Dict[str, float]]): O histórico de métricas do treinamento.
        training_duration (float): O tempo total de treinamento em segundos.
        inference_duration (float): O tempo total da inferência em segundos.
        save_path (str): O caminho do diretório onde o relatório será salvo.
    """
    final_metrics = training_history[-1]
    report_path = os.path.join(save_path, "run_summary.txt")

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
    Também salva o estado do modelo treinado.

    Args:
        model (VGAE): O modelo treinado.
        final_embeddings (torch.Tensor): Os embeddings de nós já gerados.
        wsg_obj (WSG): O objeto de dados original no formato WSG, usado como base.
        config (Config): O objeto de configuração.
        save_path (str): O caminho do diretório onde os resultados serão salvos.
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

    # a) Metadados para o novo arquivo WSG
    output_metadata = Metadata(
        dataset_name=f"{wsg_obj.metadata.dataset_name}-Embeddings",
        feature_type="dense_continuous",  # As features agora são os embeddings densos
        num_nodes=wsg_obj.metadata.num_nodes,
        num_edges=wsg_obj.metadata.num_edges,
        num_total_features=config.OUT_EMBEDDING_DIM,  # A nova "dimensão do vocabulário" é a dimensão do embedding
        processed_at=datetime.now(tz_info).isoformat(),
        directed=wsg_obj.metadata.directed,
    )

    # b) A estrutura do grafo (arestas, labels, nomes) é copiada diretamente
    output_graph_structure = wsg_obj.graph_structure

    # c) Converter os embeddings densos para o formato de features do WSG
    output_node_features = {}
    embedding_indices = list(range(config.OUT_EMBEDDING_DIM))

    for i in range(wsg_obj.metadata.num_nodes):
        node_embedding = final_embeddings[i].tolist()
        output_node_features[str(i)] = NodeFeaturesEntry(
            indices=embedding_indices,
            weights=node_embedding,
        )

    # d) Montar e validar o objeto WSG final
    output_wsg = WSG(
        metadata=output_metadata,
        graph_structure=output_graph_structure,
        node_features=output_node_features,
    )

    # --- 4. Salvar o novo arquivo .wsg.json ---
    output_filename = f"{config.DATASET_NAME}_embeddings.wsg.json"
    output_path = os.path.join(save_path, output_filename)

    # Usamos .model_dump_json() do Pydantic para serialização correta
    with open(output_path, "w") as f:
        f.write(output_wsg.model_dump_json(indent=2))

    print(f"Arquivo de embeddings no formato WSG salvo em: '{output_path}'")


def main():
    """
    Função principal que orquestra todo o processo de treinamento e
    extração de embeddings para o modelo VGAE.
    """
    # --- 1. Configuração Inicial ---
    config = Config()
    device = torch.device(config.DEVICE)
    tz_info = pytz.timezone("America/Sao_Paulo")

    print("=" * 50)
    print("INICIANDO PROCESSO DE TREINAMENTO E EXTRAÇÃO DE EMBEDDINGS")
    print("=" * 50)
    print(f"Dispositivo de treinamento: {device}")
    print(f"Dataset selecionado: {config.DATASET_NAME}")

    # --- 2. Fase 1 & 2: Pipeline de Dados (Loader -> WSG -> Converter -> PyG) ---
    print("\n[FASE 1/2] Executando pipeline de dados...")

    # Carrega os dados brutos e valida-os no formato WSG
    loader = get_loader(config.DATASET_NAME)
    wsg_obj = loader.load()

    # Converte o objeto WSG para o formato PyTorch Geometric
    pyg_data = DataConverter.to_pyg_data(wsg_obj)
    pyg_data = pyg_data.to(device)

    print("Pipeline de dados concluído. Dados prontos para o modelo.")

    # --- 3. Fase 3: Instanciação do Modelo e Otimizador ---
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

    # --- 4. Fase 4: Loop de Treinamento ---
    print("\n[FASE 4] Iniciando treinamento do modelo...")
    model, training_history = train_model(model, pyg_data, optimizer, config.EPOCHS)

    print("Treinamento finalizado.")

    # --- 5. Fase Final: Extração e Salvamento dos Resultados ---
    print("\n[FASE FINAL] Gerando e salvando resultados...")
    save_results(model, pyg_data, wsg_obj, config)

    # Salvar relatório da execução
    training_duration = sum(
        epoch_metrics["epoch"] for epoch_metrics in training_history
    )
    save_report(config, training_history, training_duration, config.RESULTS_PATH)


if __name__ == "__main__":
    main()
