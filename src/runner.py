import os
from typing import List, Dict
import psutil  # Importado
import torch   # Importado

from src.config import Config
from src.directory_manager import DirectoryManager
from src.data_format_definition import WSG
from src.classifiers import BaseClassifier


def format_bytes(b):
    """Converte bytes para um formato legível (MB ou GB)."""
    if b < 1024**3:
        return f"{b / 1024**2:.2f} MB"
    return f"{b / 1024**3:.2f} GB"


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
        
        # --- INICIAR MONITORAMENTO ---
        self.process = psutil.Process(os.getpid())
        self.mem_start = self.process.memory_info().rss
        
        # O `memory_summary` guardará o pico GERAL
        self.reports["memory_summary"] = {
            "ram_start_readable": format_bytes(self.mem_start)
        }
        # --- (NOVO) O `memory_per_model` guardará o detalhamento ---
        self.reports["memory_per_model"] = {}
        
        print(f"RAM inicial do processo de classificação: {format_bytes(self.mem_start)}")
        
        if "cuda" in config.DEVICE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(config.DEVICE)
            print("VRAM (GPU) Peak Stats zeradas.")
        

    def run(self, models_to_run: List[BaseClassifier]):
        """Executa o pipeline de treinamento e avaliação para uma lista de modelos."""

        peak_ram_during_run = self.mem_start
        peak_vram_bytes = 0
        # Mede a RAM *antes* do primeiro modelo (para medir o custo do DataConverter)
        ram_before_model = self.process.memory_info().rss 
        
        for model in models_to_run:
            print(f"\n--- 📊 Executando: {model.model_name} ---")
            
            # --- LÓGICA DE MEDIÇÃO POR MODELO ---
            acc, f1, train_time, report = model.train_and_evaluate(self.wsg_obj)
            
            # Mede a RAM *depois* que o modelo terminou
            ram_after_model = self.process.memory_info().rss
            
            # Calcula o aumento líquido causado *apenas* por este modelo
            ram_increase_bytes = ram_after_model - ram_before_model
            
            # Salva o relatório de memória para ESTE modelo
            self.reports["memory_per_model"][model.model_name] = {
                 "ram_after_model_readable": format_bytes(ram_after_model),
                 "ram_increase_readable": format_bytes(ram_increase_bytes)
            }
            print(f"--- Aumento de RAM (Líquido) para {model.model_name}: {format_bytes(ram_increase_bytes)} ---")
            
            # Atualiza a RAM "anterior" para o próximo loop
            ram_before_model = ram_after_model 
            # --- FIM DA LÓGICA POR MODELO ---

            # Atualiza o PICO GERAL
            if ram_after_model > peak_ram_during_run:
                peak_ram_during_run = ram_after_model
            
            if "cuda" in self.config.DEVICE and torch.cuda.is_available():
                current_vram_peak = torch.cuda.max_memory_allocated(self.config.DEVICE)
                if current_vram_peak > peak_vram_bytes:
                    peak_vram_bytes = current_vram_peak

            self.results[model.model_name] = {
                "accuracy": acc,
                "f1_score_weighted": f1,
                "training_time_seconds": train_time,
            }
            if report:
                self.reports[f"{model.model_name}_classification_report"] = report

        # --- FINALIZAR E SALVAR MÉTRICAS GERAIS ---
        mem_end_run = self.process.memory_info().rss
        self.reports["memory_summary"].update({
            "ram_end_readable": format_bytes(mem_end_run),
            "ram_peak_during_run_readable": format_bytes(peak_ram_during_run),
            "vram_peak_readable": format_bytes(peak_vram_bytes)
        })
        print(f"\n--- Resumo do Runner ---")
        print(f"PICO de RAM (Geral): {format_bytes(peak_ram_during_run)}")
        print(f"PICO de VRAM (Geral): {format_bytes(peak_vram_bytes)}")
        # --- FIM ---

        # Salva o relatório (que agora contém 'memory_summary' E 'memory_per_model')
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