import os
import time
import threading
import psutil
import torch
from typing import Callable, Any, Tuple


class MemoryTracker:
    """
    Um rastreador que monitora o pico de uso de memória RAM e VRAM
    durante a execução de uma função.
    """

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.peak_ram_usage = 0
        self.peak_vram_usage = 0
        self.keep_monitoring = True
        self.monitoring_thread = None

    def _monitor(self):
        """Função executada em uma thread separada para monitorar o uso de memória."""
        self.peak_ram_usage = self.process.memory_info().rss  # RAM inicial

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.peak_vram_usage = torch.cuda.max_memory_allocated()

        while self.keep_monitoring:
            # Monitora RAM
            self.peak_ram_usage = max(
                self.peak_ram_usage, self.process.memory_info().rss
            )

            # Monitora VRAM (usando a função de pico do PyTorch)
            if torch.cuda.is_available():
                self.peak_vram_usage = max(
                    self.peak_vram_usage, torch.cuda.max_memory_allocated()
                )

            time.sleep(0.1)  # Monitora a cada 100ms

    def track(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """
        Executa uma função enquanto monitora o uso de memória em segundo plano.

        Args:
            func (Callable): A função a ser executada (ex: model.train_and_evaluate).
            *args, **kwargs: Argumentos para a função.

        Returns:
            Tuple[Any, Dict[str, float]]: O resultado da função e um dicionário com as métricas de memória.
        """
        self.keep_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitoring_thread.start()

        try:
            # Executa a função principal
            result = func(*args, **kwargs)
        finally:
            # Para o monitoramento
            self.keep_monitoring = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=1.0)

        # Converte para Megabytes
        peak_ram_mb = self.peak_ram_usage / (1024 * 1024)
        peak_vram_mb = self.peak_vram_usage / (1024 * 1024)

        memory_metrics = {
            "peak_ram_mb": peak_ram_mb,
            "peak_vram_mb": peak_vram_mb if torch.cuda.is_available() else 0.0,
        }

        return result, memory_metrics
