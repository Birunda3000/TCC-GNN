import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data_converter import DataConverter
from src.data_format_definition import WSG
from src.config import Config


def _get_dense_features_from_wsg(wsg_obj: WSG, config: Config, device: torch.device):
    """
    Função interna para converter um WSG em um tensor de features denso e labels.
    Esta função encapsula toda a lógica de conversão de dados.
    """
    pyg_data = DataConverter.to_pyg_data(wsg_obj)
    feature_type = wsg_obj.metadata.feature_type

    if feature_type == "sparse_binary":
        print(
            "Detectado feature_type 'sparse_binary'. Usando EmbeddingBag para densificar."
        )
        embedder = nn.EmbeddingBag(
            pyg_data.num_total_features, config.EMBEDDING_DIM, mode="sum"
        ).to(device)
        dense_features = embedder(
            pyg_data.feature_indices.to(device),
            pyg_data.feature_offsets.to(device),
            pyg_data.feature_weights.to(device),
        ).detach()

    elif feature_type == "dense_continuous":
        print(
            "Detectado feature_type 'dense_continuous'. Reconstruindo tensor de embeddings."
        )
        num_nodes, embedding_dim = (
            wsg_obj.metadata.num_nodes,
            wsg_obj.metadata.num_total_features,
        )
        dense_features = torch.zeros((num_nodes, embedding_dim), device=device)
        for i in range(len(pyg_data.feature_offsets) - 1):
            start, end = pyg_data.feature_offsets[i], pyg_data.feature_offsets[i + 1]
            dense_features[i, pyg_data.feature_indices[start:end]] = (
                pyg_data.feature_weights[start:end]
            )
    else:
        raise ValueError(f"Tipo de feature não suportado: {feature_type}")

    return dense_features, pyg_data.y.to(device), pyg_data.edge_index.to(device)


def train_and_evaluate_sklearn_model(model, wsg_obj: WSG, config: Config):
    """
    Recebe um WSG, converte os dados, treina e avalia um modelo Scikit-learn.
    """
    device = torch.device(config.DEVICE)
    X_features, y_labels, _ = _get_dense_features_from_wsg(wsg_obj, config, device)

    valid_indices = (y_labels != -1).nonzero(as_tuple=True)[0].cpu().numpy()
    X_filtered = X_features[valid_indices].cpu().numpy()
    y_filtered = y_labels[valid_indices].cpu().numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered,
        y_filtered,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=y_filtered,
    )

    model_name = model.__class__.__name__
    print(f"Treinando {model_name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Resultados: Acurácia={acc:.4f}, F1={f1:.4f}, Tempo={train_time:.2f}s")
    return acc, f1, train_time


def run_pytorch_classification(
    model, wsg_obj: WSG, config: Config, optimizer_class, criterion, use_gnn: bool
):
    """
    Recebe um WSG, converte, treina e avalia um modelo PyTorch (MLP, GCN, ou GAT).
    """
    device = torch.device(config.DEVICE)
    model.to(device)

    # 1. Conversão de dados
    X_features, y_labels, edge_index = _get_dense_features_from_wsg(
        wsg_obj, config, device
    )

    # 2. Preparação de dados para treino
    valid_indices = (y_labels != -1).nonzero(as_tuple=True)[0]
    train_indices, test_indices = train_test_split(
        valid_indices.cpu().numpy(),
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=y_labels[valid_indices].cpu().numpy(),
    )
    train_mask = torch.zeros_like(y_labels, dtype=torch.bool, device=device)
    test_mask = torch.zeros_like(y_labels, dtype=torch.bool, device=device)
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    optimizer = optimizer_class(model.parameters(), lr=0.01)

    # 3. Loop de Treinamento
    print(f"Iniciando treinamento do modelo {model.__class__.__name__}...")
    start_time = time.time()
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()

        # O forward pass decide o que usar
        out = model(X_features, edge_index) if use_gnn else model(X_features)

        loss = criterion(out[train_mask], y_labels[train_mask])
        loss.backward()
        optimizer.step()
    train_time = time.time() - start_time
    print(f"Treinamento concluído em {train_time:.2f}s.")

    # 4. Avaliação
    model.eval()
    with torch.no_grad():
        out = model(X_features, edge_index) if use_gnn else model(X_features)
        preds = out[test_mask].argmax(dim=1)
        labels = y_labels[test_mask]

        acc = accuracy_score(labels.cpu(), preds.cpu())
        f1 = f1_score(labels.cpu(), preds.cpu(), average="weighted", zero_division=0)
        report = classification_report(labels.cpu(), preds.cpu(), zero_division=0)

    print(f"Resultados Finais: Acurácia={acc:.4f}, F1={f1:.4f}")
    return acc, f1, train_time, report
