import pandas as pd
import os

# --- Configuração dos Caminhos ---
# Caminhos dos arquivos originais
CONTENT_FILE = "data/datasets/cora/cora.content"
CITES_FILE = "data/datasets/cora/cora.cites"

# Caminho para salvar os novos arquivos CSV
OUTPUT_DIR = "data/processed/cora"
OUTPUT_CONTENT_CSV = os.path.join(OUTPUT_DIR, "cora_content.csv")
OUTPUT_CITES_CSV = os.path.join(OUTPUT_DIR, "cora_cites.csv")


# --- Função para converter o arquivo .content ---
def convert_content():
    """
    Lê o arquivo cora.content e o converte para um formato CSV com cabeçalhos,
    com as colunas 'paper_id' e 'class_label' primeiro.
    """
    print(f"Lendo arquivo de conteúdo de: {CONTENT_FILE}")

    # Lê o arquivo, que é separado por tabulação e não tem cabeçalho.
    # Trata a primeira coluna (ID) como string para evitar problemas.
    df_content = pd.read_csv(CONTENT_FILE, sep="\t", header=None, dtype={0: str})

    # Define os nomes das colunas
    num_features = df_content.shape[1] - 2
    feature_cols = [f"word_{i}" for i in range(num_features)]

    # Atribui nomes temporários na ordem original de leitura
    df_content.columns = ["paper_id"] + feature_cols + ["class_label"]

    # --- ALTERAÇÃO: Reordena as colunas ---
    # Cria a nova ordem de colunas desejada
    new_column_order = ["paper_id", "class_label"] + feature_cols

    # Aplica a nova ordem ao DataFrame
    df_content = df_content[new_column_order]

    print(
        f"Convertendo para CSV com as colunas 'paper_id' e 'class_label' no início..."
    )
    df_content.to_csv(OUTPUT_CONTENT_CSV, index=False)
    print(f"Arquivo 'cora_content.csv' salvo com sucesso em: {OUTPUT_CONTENT_CSV}")


# --- Função para converter o arquivo .cites ---
def convert_cites():
    """
    Lê o arquivo cora.cites e o converte para um formato CSV com cabeçalhos.
    """
    print(f"\nLendo arquivo de citações de: {CITES_FILE}")

    # Lê o arquivo, tratando ambos os IDs como strings.
    df_cites = pd.read_csv(CITES_FILE, sep="\t", header=None, dtype=str)

    # De acordo com o README: a primeira coluna é o artigo CITADO, a segunda é o que CITA.
    df_cites.columns = ["cited_paper_id", "citing_paper_id"]

    print("Convertendo para CSV...")
    df_cites.to_csv(OUTPUT_CITES_CSV, index=False)
    print(f"Arquivo 'cora_cites.csv' salvo com sucesso em: {OUTPUT_CITES_CSV}")


# --- Execução Principal ---
if __name__ == "__main__":
    # Cria o diretório de saída se ele não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    convert_content()
    convert_cites()

    print("\nConversão concluída!")
