import os
import json
from typing import List, Dict

from src.config import Config
from src.directory_manager import DirectoryManager
from src.data_format_definition import WSG
from src.classifiers import BaseClassifier


class ExperimentRunner:
    """
    Encapsula a lógica de execução de um experimento de classificação,
    unificando o treinamento de modelos, a coleta de resultados e o salvamento de relatórios.
    """

    def __init__(
        self,
        config: Config,
        run_folder_name: str,
        wsg_obj: WSG,
        experiment_name: str,
    ):
        self.config = config
        self.wsg_obj = wsg_obj
        self.experiment_name = experiment_name
        self.data_source_name = wsg_obj.metadata.dataset_name

        self.directory_manager = DirectoryManager(
            timestamp=config.TIMESTAMP, run_folder_name=run_folder_name
        )
        self.run_path = self.directory_manager.get_run_path()
        self.results: Dict = {}
        self.reports: Dict = {}

    def run(self, models_to_run: List[BaseClassifier]):
        """
        Executa o pipeline de treinamento e avaliação para uma lista de modelos.
        """
        print(f"Iniciando experimento: {self.experiment_name}")
        print(f"Fonte dos dados: {self.data_source_name}")
        print("-" * 65)

        for model in models_to_run:
            acc, f1, train_time, report = model.train_and_evaluate(self.wsg_obj)
            self.results[model.model_name] = {
                "accuracy": acc,
                "f1_score_weighted": f1,
                "training_time_seconds": train_time,
            }
            if report:
                self.reports[f"{model.model_name}_classification_report"] = report

        self._save_report()
        self._print_summary()
        self._finalize_directory()

    def _save_report(self):
        """Salva um relatório consolidado em formato JSON."""
        summary = {
            "experiment_name": self.experiment_name,
            "data_source": self.data_source_name,
            "classification_results": self.results,
            "detailed_reports": self.reports,
        }
        report_path = os.path.join(self.run_path, "classification_summary.json")
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"\nRelatório consolidado salvo em: '{report_path}'")

    def _print_summary(self):
        """Imprime a tabela de resumo dos resultados no console."""
        print("\n" + "=" * 65)
        print(f"RELATÓRIO: {self.experiment_name}".center(65))
        print("-" * 65)
        print(f"Fonte dos Dados: {self.data_source_name}")
        print("-" * 65)
        print(
            f"{'Modelo':<25} | {'Acurácia':<12} | {'F1-Score':<12} | {'Tempo (s)':<10}"
        )
        print("=" * 65)
        for name, metrics in self.results.items():
            print(
                f"{name:<25} | {metrics['accuracy']:<12.4f} | {metrics['f1_score_weighted']:<12.4f} | {metrics['training_time_seconds']:<10.2f}"
            )
        print("=" * 65)

    def _finalize_directory(self):
        """Finaliza e renomeia o diretório de execução."""
        if not self.results:
            print("Nenhum resultado para finalizar o diretório.")
            return

        best_model = max(self.results.items(), key=lambda x: x[1]["accuracy"])
        best_acc = best_model[1]["accuracy"]
        best_model_name = best_model[0].lower().replace("classifier", "")

        final_path = self.directory_manager.finalize_run_directory(
            dataset_name=self.data_source_name,
            metrics={"best_acc": best_acc, "model": best_model_name},
        )
        print(f"\nProcesso concluído! Resultados salvos em: '{final_path}'")
