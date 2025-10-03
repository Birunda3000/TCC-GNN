from src.data_loader import DirectWSGLoader
from src.config import Config
import torch
import numpy as np
import random
from dp_anonynizer import TabularLaplaceAnonymizer
import pandas as pd
from src.data_format_definition import WSG


def wsg_nodes_to_dataframe(wsg_obj: WSG) -> pd.DataFrame:
    """
    Converte as informações de nós de um objeto WSG para um DataFrame do pandas,
    com uma coluna para cada dimensão da feature.

    Args:
        wsg_obj: O objeto WSG contendo os dados do grafo.

    Returns:
        Um pandas DataFrame com as informações e features dos nós.
    """
    node_data = []
    # O número total de features define quantas colunas de feature teremos.
    num_features = wsg_obj.metadata.num_total_features
    feature_col_names = [f"feature_{j}" for j in range(num_features)]

    for i in range(wsg_obj.metadata.num_nodes):
        node_id_str = str(i)

        # Informações base do nó
        current_node_info = {
            "node_id": i,
            "node_name": wsg_obj.graph_structure.node_names[i],
            "label": wsg_obj.graph_structure.y[i],
        }

        # Extrai as features esparsas para o nó atual
        node_features_entry = wsg_obj.node_features.get(node_id_str)

        # Cria um vetor denso de features para o nó atual
        dense_features = np.zeros(num_features)
        if node_features_entry:
            # Preenche o vetor denso com os pesos das features existentes
            dense_features[node_features_entry.indices] = node_features_entry.weights

        # Adiciona as features densas ao dicionário do nó, uma por coluna
        current_node_info.update(dict(zip(feature_col_names, dense_features)))

        node_data.append(current_node_info)

    return pd.DataFrame(node_data)


config = Config()
torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)

input_file_path = "data/output/EMBEDDING_RUNS/musae-github__loss_0_9487__emb_dim_64__01-10-2025_15-30-48/musae-github_embeddings.wsg.json"

loader = DirectWSGLoader(file_path=input_file_path)
wsg_obj = loader.load()


# Exemplo de como usar a função:
if __name__ == "__main__":
    nodes_df = wsg_nodes_to_dataframe(wsg_obj)
    print("DataFrame gerado a partir do WSG (com colunas de features):")
    # Mostra as primeiras 5 linhas e as primeiras 10 colunas para melhor visualização
    print(nodes_df.iloc[:, :10].head(10))
    print("\nInformações do DataFrame:")

    numeric_columns = nodes_df.select_dtypes(include=np.number).columns.tolist()

    # CORREÇÃO: Usa-se o método .remove() para remover itens de uma lista.
    # É bom verificar se o item existe antes de tentar removê-lo.
    if "node_id" in numeric_columns:
        numeric_columns.remove("node_id")

    if "label" in numeric_columns:
        numeric_columns.remove("label")

    # A coluna "node_name" não é numérica e não estará na lista, então não precisa ser removida.

    bounds_dict = {
        col: (nodes_df[col].min(), nodes_df[col].max()) for col in numeric_columns
    }
    print("\nColunas Numéricas para Anonimização:", numeric_columns[:5], "...")

    anonymizer = TabularLaplaceAnonymizer(
        epsilon=0.01,
        numeric_columns=numeric_columns,
        bounds=bounds_dict,
        seed=config.RANDOM_SEED,
    )

    anonymized_df = anonymizer.fit(nodes_df).transform(nodes_df)
    print("--------------------------------------------------------")
    print("\nDataFrame Anonimizado (com colunas de features):")
    print(anonymized_df.iloc[:, :10].head(10))
    print("\nInformações do DataFrame Anonimizado:")
    anonymized_df.info()

    # --- INÍCIO: TESTE DE CLASSIFICAÇÃO PARA COMPARAÇÃO ---
    from sklearn.model_selection import train_test_split

    # from sklearn.linear_model import LogisticRegression # Não é mais necessário
    from sklearn.neighbors import KNeighborsClassifier  # Importa o KNN
    from sklearn.metrics import accuracy_score, f1_score

    def train_and_evaluate_classifier(df, feature_cols, target_col, dataset_name):
        """
        Função auxiliar para treinar e avaliar um modelo KNN.
        """
        print(f"\n--- Treinando e Avaliando no Dataset: {dataset_name} ---")

        # Define features (X) e target (y)
        X = df[feature_cols]
        y = df[target_col]

        # Divide os dados em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=config.RANDOM_SEED, stratify=y
        )

        # Inicializa e treina o modelo KNN
        # O KNN não usa random_state, mas é determinístico por natureza.
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)

        # Faz predições e avalia a performance
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"Acurácia: {accuracy:.4f}")
        print(f"F1-Score (Weighted): {f1:.4f}")
        return accuracy, f1

    print("\n" + "=" * 50)
    print("COMPARAÇÃO DE PERFORMANCE DE CLASSIFICAÇÃO (UTILIDADE)")
    print("=" * 50)

    # A coluna alvo para a classificação é 'label'
    target_column = "label"

    # Treina e avalia no DataFrame original
    acc_orig, f1_orig = train_and_evaluate_classifier(
        df=nodes_df,
        feature_cols=numeric_columns,  # Usando as mesmas colunas de features
        target_col=target_column,
        dataset_name="Original",
    )

    # Treina e avalia no DataFrame anonimizado
    acc_anon, f1_anon = train_and_evaluate_classifier(
        df=anonymized_df,
        feature_cols=numeric_columns,
        target_col=target_column,
        dataset_name="Anonimizado",
    )

    print("\n--- Resumo da Comparação ---")
    print(f"Queda na Acurácia: {acc_orig - acc_anon:.4f}")
    print(f"Queda no F1-Score: {f1_orig - f1_anon:.4f}")
    print("=" * 50)
    # --- FIM: TESTE DE CLASSIFICAÇÃO ---
