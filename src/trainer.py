import torch
import torch.nn.functional as F
from typing import Tuple

@torch.no_grad()
def _get_accuracy(out: torch.Tensor, data: 'Data', mask: torch.Tensor) -> float:
    """Calcula a acurácia para uma dada máscara."""
    pred = out.argmax(dim=1)
    correct = pred[mask] == data.y[mask]
    return int(correct.sum()) / int(mask.sum())

def train_step(model: torch.nn.Module, data: 'Data', optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    """
    Executa um passo de treinamento (uma época) na máscara de treino.
    
    Retorna:
        Tuple[float, float]: A perda (loss) e a acurácia de treino.
    """
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    
    train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    train_loss.backward()
    optimizer.step()
    
    train_acc = _get_accuracy(out, data, data.train_mask)
    
    return train_loss.item(), train_acc

@torch.no_grad()
def eval_step(model: torch.nn.Module, data: 'Data') -> Tuple[float, float]:
    """
    Executa um passo de avaliação na máscara de validação.

    Retorna:
        Tuple[float, float]: A perda (loss) e a acurácia de validação.
    """
    model.eval()
    
    out = model(data.x, data.edge_index)
    
    val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
    val_acc = _get_accuracy(out, data, data.val_mask)
    
    return val_loss.item(), val_acc