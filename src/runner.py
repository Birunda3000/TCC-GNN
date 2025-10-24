import os
from typing import List, Dict

from src.config import Config
from src.directory_manager import DirectoryManager
from src.data_format_definition import WSG
from src.classifiers import BaseClassifier


class ExperimentRunner:
    """Orquestra a execução de um experimento de classificação."""

    def __init__(
        self, config: Config, run_folder_name: str, wsg_obj: WSG, data_source_name: str
    ):
        self.config = config
        self.wsg_obj = wsg_obj
        self.data_source_name = data_source_name
        self.directory_manager = DirectoryManager(config.TIMESTAMP, run_folder_name)
        self.results: Dict = {}
        self.reports: Dict = {}

    def run(self, models_to_run: List[BaseClassifier]):
        """Executa o pipeline de treinamento e avaliação para uma lista de modelos."""
        for model in models_to_run:
            acc, f1, train_time, report = model.train_and_evaluate(self.wsg_obj)
            self.results[model.model_name] = {
                "accuracy": acc,
                "f1_score_weighted": f1,
                "training_time_seconds": train_time,
            }
            if report:
                self.reports[f"{model.model_name}_classification_report"] = report

        # Usa os métodos do DirectoryManager
        self.directory_manager.save_classification_report(
            input_file=self.data_source_name, results=self.results, reports=self.reports
        )
        self.directory_manager.print_summary_table(
            results=self.results,
            input_file_path=self.data_source_name,
            feature_type=self.wsg_obj.metadata.feature_type,
        )

        # Finaliza o diretório
        if self.results:
            best_model = max(self.results.items(), key=lambda x: x[1]["accuracy"])
            best_acc = best_model[1]["accuracy"]
            best_model_name = best_model[0].lower().replace("classifier", "")
            final_path = self.directory_manager.finalize_run_directory(
                dataset_name=self.wsg_obj.metadata.dataset_name,
                metrics={"best_acc": best_acc, "model": best_model_name},
            )
            print(f"\nProcesso concluído! Resultados salvos em: '{final_path}'")
