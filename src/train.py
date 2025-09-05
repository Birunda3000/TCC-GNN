import torch
from tqdm import tqdm
import pandas as pd
import os
import random  # Importa o módulo random

from config import Config
from data_loader import get_loader
from model import GCN
from trainer import eval_step, train_step
from utils import create_deterministic_masks, set_seeds
from directory_manager import TrainingRunManager


def run_training():
    """
    Orquestra o carregamento dos dados, a configuração do modelo e o loop de treinamento.
    """
    # Gera uma semente aleatória para esta execução e a define.
    seed = random.randint(0, 2**32 - 1)
    set_seeds(seed)

    # 0. Iniciar o gerenciador de diretórios
    run_manager = TrainingRunManager(
        base_output_dir="data/output", dataset_name=Config.DATASET_NAME
    )
    output_path = run_manager.get_run_path()

    # Salva os parâmetros da execução, incluindo a semente, em um arquivo de texto.
    params_path = os.path.join(output_path, "run_parameters.txt")
    with open(params_path, "w") as f:
        f.write(f"seed: {seed}\n")
        f.write(f"dataset: {Config.DATASET_NAME}\n")
        f.write(f"split_ratio: {Config.SPLIT_RATIO}\n")
        f.write(f"device: {Config.DEVICE}\n")
    print(f"Parâmetros da execução salvos em: '{params_path}'")

    # 1. Carregar dados
    loader = get_loader(Config.DATASET_NAME)
    data, num_classes, _ = loader.load()

    # 2. Criar máscaras de divisão e mover para o dispositivo
    data = create_deterministic_masks(data, Config.SPLIT_RATIO)
    data = data.to(Config.DEVICE)

    # 3. Instanciar modelo e otimizador
    model = GCN(
        in_channels=data.num_node_features, hidden_channels=16, out_channels=num_classes
    ).to(Config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # 4. Loop de Treinamento
    print("\nIniciando o treinamento...")
    history = []
    best_val_acc = 0.0
    model_path = os.path.join(output_path, "best_model.pth")

    pbar = tqdm(range(1, 201), desc="Treinando")
    for epoch in pbar:
        train_loss, train_acc = train_step(model, data, optimizer)
        val_loss, val_acc = eval_step(model, data)

        # Atualiza a melhor acurácia de validação e salva o melhor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(model.state_dict(), model_path)
            # O postfix será atualizado na próxima linha, mas podemos adicionar um log se quisermos
            # print(f"Novo melhor modelo salvo em época {epoch} com Val Acc: {val_acc:.4f}")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        pbar.set_postfix(
            {
                "Train Loss": f"{train_loss:.4f}",
                "Val Acc": f"{val_acc:.4f}",
                "Best Val": f"{best_val_acc:.4f}",
            }
        )

    print("\nTreinamento concluído!")

    # 5. Avaliação Final no conjunto de teste
    model.eval()
    # Carrega o melhor modelo salvo para a avaliação final
    model.load_state_dict(torch.load(model_path))

    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())

    print(f"Acurácia Final no Conjunto de Teste: {test_acc:.4f}")
    print(f"Melhor modelo salvo em: '{model_path}'")

    # 6. Salvar artefatos e finalizar a execução
    df_history = pd.DataFrame(history)
    history_path = os.path.join(output_path, "training_history.csv")
    df_history.to_csv(history_path, index=False)
    print(f"Histórico de treinamento salvo em: '{history_path}'")

    # Renomeia o diretório com a métrica final
    final_dir_path = run_manager.finalize(best_val_acc)

    return df_history, final_dir_path


if __name__ == "__main__":
    training_history, final_path = run_training()

    if final_path:
        print(f"\nExecução finalizada. Artefatos salvos em: '{final_path}'")
