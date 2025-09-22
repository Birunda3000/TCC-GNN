# train.py

import torch
import torch.optim as optim
import os
import json
from datetime import datetime
import pytz

# --- Importa nossos módulos customizados ---
from src.config import Config
from src.data_loader import get_loader
from src.data_converter import DataConverter
from src.model import VGAE
from src.data_format_definition import WSG
from torch_geometric.data import Data


def train_model(
    model: VGAE, data: Data, optimizer: optim.Optimizer, epochs: int
) -> VGAE:
    """
    Executa o loop de treinamento para o modelo VGAE.

    Args:
        model (VGAE): A instância do modelo a ser treinado.
        data (Data): Os dados do grafo no formato PyG.
        optimizer (optim.Optimizer): O otimizador.
        epochs (int): O número de épocas para treinar.

    Returns:
        VGAE: O modelo treinado.
    """
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

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch:03d} | Loss Total: {total_loss:.4f} | "
                f"Reconstrução: {recon_loss:.4f} | KL: {kl_loss:.4f}"
            )
    return model


def save_results(model: VGAE, pyg_data: Data, wsg_obj: WSG, config: Config):
    """
    Gera e salva os embeddings finais, o modelo e os metadados.

    Args:
        model (VGAE): O modelo treinado.
        pyg_data (Data): Os dados do grafo no formato PyG.
        wsg_obj (WSG): O objeto de dados original no formato WSG.
        config (Config): O objeto de configuração.
    """
    model.eval()
    tz_info = pytz.timezone("America/Sao_Paulo")

    # Gerar os embeddings finais
    final_embeddings = model.get_embeddings(pyg_data)

    # Salvar o estado do modelo treinado
    model_path = os.path.join(config.OUTPUT_PATH, f"vgae_model_{config.TIMESTAMP}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Modelo treinado salvo em: '{model_path}'")

    # Construir o "dossiê" completo de saída em JSON

    # 1. Pré-calcular informações estruturais para acesso rápido
    adj = {i: [] for i in range(wsg_obj.metadata.num_nodes)}
    for i in range(len(wsg_obj.graph_structure.edge_index[0])):
        u = wsg_obj.graph_structure.edge_index[0][i]
        v = wsg_obj.graph_structure.edge_index[1][i]
        adj[u].append(v)

    # 2. Montar o dicionário de saída
    output_data = {
        "metadata": {
            "dataset_name": wsg_obj.metadata.dataset_name,
            "model_name": "VGAE",
            "embedding_dim": config.OUT_EMBEDDING_DIM,
            "num_nodes": wsg_obj.metadata.num_nodes,
            "num_edges": wsg_obj.metadata.num_edges,
            "num_total_features": wsg_obj.metadata.num_total_features,
            "directed": wsg_obj.metadata.directed,
            "training_timestamp": datetime.now(tz_info).isoformat(),
            "source_model_path": model_path,
        },
        "nodes": {},
    }

    for i in range(wsg_obj.metadata.num_nodes):
        node_id_str = str(i)
        output_data["nodes"][node_id_str] = {
            "node_id": i,
            "original_name": wsg_obj.graph_structure.node_names[i],
            "embedding": final_embeddings[i]
            .cpu()
            .tolist(),  # Mover para CPU e converter
            "original_features": wsg_obj.node_features[node_id_str].dict(),
            "structural_info": {"degree": len(adj[i]), "neighbors": adj[i]},
        }

    # 3. Salvar o arquivo JSON final
    output_filename = f"embeddings_output_{config.TIMESTAMP}.json"
    output_path = os.path.join(config.OUTPUT_PATH, output_filename)

    with open(output_path, "w") as f:
        json.dump(output_data, f)  # Sem indentação para economizar espaço

    print(f"Arquivo de saída de embeddings salvo em: '{output_path}'")
    print("\n" + "=" * 50)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print("=" * 50)


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
    train_model(model, pyg_data, optimizer, config.EPOCHS)

    print("Treinamento finalizado.")

    # --- 5. Fase Final: Extração e Salvamento dos Resultados ---
    print("\n[FASE FINAL] Gerando e salvando resultados...")
    save_results(model, pyg_data, wsg_obj, config)


if __name__ == "__main__":
    main()
