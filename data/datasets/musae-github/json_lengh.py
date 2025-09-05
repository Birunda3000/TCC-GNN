import json

def check_json_lengths(file_path):
    """
    Lê um arquivo JSON e verifica se todos os arrays têm o mesmo tamanho.

    Args:
      file_path: O caminho para o arquivo JSON.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Pega um iterador para os valores no dicionário
    iterator = iter(data.values())
    try:
        first_length = len(next(iterator))
    except StopIteration:
        print("O JSON está vazio.")
        return

    # Verifica o comprimento do resto dos arrays
    for key, value in data.items():
        if len(value) != first_length:
            print(f"Os arrays no JSON não têm todos o mesmo tamanho.")
            print(f"O primeiro array ('0') tem {first_length} elementos.")
            print(f"O array '{key}' tem {len(value)} elementos.")
            return

    print("Todos os arrays no JSON têm o mesmo tamanho.")

# Exemplo de como usar a função:
check_json_lengths('musae_git_features.json')