import os
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo  # Importa a biblioteca para fusos horários


class TrainingRunManager:
    """
    Gerencia a criação e o versionamento do diretório de saída para uma execução de treino.

    Cria um diretório temporário no início e o renomeia com a métrica final
    após a conclusão.
    """

    def __init__(self, base_output_dir: str, dataset_name: str):
        """
        Inicializa o gerenciador e cria o diretório temporário.

        Args:
            base_output_dir (str): O diretório base onde os resultados serão salvos (ex: 'data/output').
            dataset_name (str): O nome do dataset sendo usado.
        """
        self.base_dir = base_output_dir
        self.dataset_name = dataset_name

        # CORREÇÃO: Converte a hora UTC do container para o fuso horário de São Paulo.
        utc_now = datetime.now(ZoneInfo("UTC"))
        local_now = utc_now.astimezone(ZoneInfo("America/Sao_Paulo"))
        self.timestamp = local_now.strftime("%d-%m-%Y_%H-%M-%S")

        tmp_dir_name = f"_tmp_{self.dataset_name}__{self.timestamp}"
        self.run_path = os.path.join(self.base_dir, tmp_dir_name)

        os.makedirs(self.run_path, exist_ok=True)
        print(f"Diretório de execução temporário criado em: '{self.run_path}'")

    def get_run_path(self) -> str:
        """Retorna o caminho para o diretório da execução atual."""
        return self.run_path

    def finalize(self, best_val_acc: float) -> str:
        """
        Renomeia o diretório temporário para o nome final, incluindo a acurácia.

        Args:
            best_val_acc (float): A melhor acurácia de validação alcançada.

        Returns:
            str: O caminho final do diretório.
        """
        if not os.path.exists(self.run_path):
            print(
                f"Aviso: Diretório temporário '{self.run_path}' não encontrado para finalizar."
            )
            return ""

        # Usa o timestamp da instância para garantir consistência.
        final_dir_name = (
            f"{self.dataset_name}__val_acc_{best_val_acc:.4f}__{self.timestamp}"
        )
        final_path = os.path.join(self.base_dir, final_dir_name)

        os.rename(self.run_path, final_path)

        self.run_path = final_path
        return final_path
