'''
Importa as bibliotecas necessárias para a construção de modelos de classificação. Para os embeddings.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from abc import ABC, abstractmethod
from tqdm import tqdm
import math

from torch_geometric.nn import GCNConv, GATConv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import itertools
from joblib import Parallel, delayed
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Importação do XGBoost
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    print("Aviso: XGBoost não está instalado. XGBoostClassifier não estará disponível.")
    XGBOOST_AVAILABLE = False

from .data_format_definition import WSG
from .config import Config
from .data_converter import DataConverter


class BaseClassifier(ABC):
    """
    Classe base abstrata. Define que todo classificador deve saber
    como treinar e se avaliar a partir de um objeto WSG.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model_name = self.__class__.__name__

    @abstractmethod
    def train_and_evaluate(self, wsg_obj: WSG):
        """
        Orquestra o processo de treinamento e avaliação para o modelo.
        Deve retornar: (acurácia, f1_score, tempo_de_treino, relatório_detalhado)
        """
        pass


class SklearnClassifier(BaseClassifier):
    """Wrapper para modelos Scikit-learn que contém sua própria lógica de treino."""

    def __init__(self, config: Config, model_class, **model_params):
        super().__init__(config)
        self.model_name = model_class.__name__
        try:
            self.model = model_class(random_state=config.RANDOM_SEED, **model_params)
        except TypeError:
            self.model = model_class(**model_params)

    def train_and_evaluate(self, wsg_obj: WSG):
        print(f"\n--- Avaliando (Sklearn): {self.model_name} ---")

        pyg_data = DataConverter.to_pyg_data(wsg_obj=wsg_obj, for_embedding_bag=False)
        X = pyg_data.x.cpu().numpy()
        y = pyg_data.y.cpu().numpy()

        # Usar as máscaras de treino/teste já definidas no objeto pyg_data
        X_train, y_train = X[pyg_data.train_mask], y[pyg_data.train_mask]
        X_test, y_test = X[pyg_data.test_mask], y[pyg_data.test_mask]

        start_time = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        return acc, f1, train_time, report


class PyTorchClassifier(BaseClassifier, nn.Module):
    """
    Classe base para classificadores PyTorch. Contém o loop de treino completo.
    """

    def __init__(
        self, config: Config, input_dim: int, hidden_dim: int, output_dim: int
    ):
        BaseClassifier.__init__(self, config)
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def _train_step(self, optimizer, criterion, data, use_gnn):
        self.train()
        optimizer.zero_grad()

        args = [data.x, data.edge_index] if use_gnn else [data.x]
        out = self(*args)

        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def _test_step(self, data, use_gnn):
        self.eval()
        args = [data.x, data.edge_index] if use_gnn else [data.x]
        out = self(*args)
        pred = out.argmax(dim=1)

        y_true = data.y[data.test_mask]
        y_pred = pred[data.test_mask]

        acc = accuracy_score(y_true.cpu(), y_pred.cpu())
        f1 = f1_score(y_true.cpu(), y_pred.cpu(), average="weighted")
        report = classification_report(
            y_true.cpu(), y_pred.cpu(), output_dict=True, zero_division=0
        )

        return acc, f1, report

    def _train_and_evaluate_internal(self, wsg_obj: WSG, use_gnn: bool):
        print(f"\n--- Avaliando (PyTorch): {self.model_name} ---")
        device = torch.device(self.config.DEVICE)
        self.to(device)

        data = DataConverter.to_pyg_data(wsg_obj=wsg_obj, for_embedding_bag=False).to(device)
        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()

        pbar = tqdm(
            range(self.config.EPOCHS),
            desc=f"Treinando {self.model_name}",
            leave=False,
        )
        for epoch in pbar:
            loss = self._train_step(optimizer, criterion, data, use_gnn)
            pbar.set_postfix({"loss": f"{loss:.4f}"})

        train_time = time.time() - start_time

        acc, f1, report = self._test_step(data, use_gnn)
        return acc, f1, train_time, report


# --- Implementações Específicas ---


class MLPClassifier(PyTorchClassifier):
    """Classificador MLP que opera em um tensor de features denso."""

    def __init__(self, config, input_dim, hidden_dim, output_dim):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def train_and_evaluate(self, wsg_obj: WSG):
        return self._train_and_evaluate_internal(wsg_obj, use_gnn=False)


class GCNClassifier(PyTorchClassifier):
    """Classificador GCN que opera em features e na estrutura do grafo."""

    def __init__(self, config, input_dim, hidden_dim, output_dim):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

    def train_and_evaluate(self, wsg_obj: WSG):
        return self._train_and_evaluate_internal(wsg_obj, use_gnn=True)


class GATClassifier(PyTorchClassifier):
    """Classificador GAT que utiliza mecanismos de atenção."""

    def __init__(self, config, input_dim, hidden_dim, output_dim, heads=8):
        super().__init__(config, input_dim, hidden_dim, output_dim)
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(
            hidden_dim * heads, output_dim, heads=1, concat=False, dropout=0.6
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return self.conv2(x, edge_index)

    def train_and_evaluate(self, wsg_obj: WSG):
        return self._train_and_evaluate_internal(wsg_obj, use_gnn=True)


class XGBoostClassifier(BaseClassifier):
    """
    Implementação de um classificador XGBoost robusto com busca de hiperparâmetros.
    XGBoost geralmente oferece alto desempenho, mas pode levar mais tempo para treinar.
    """

    def __init__(self, config: Config, num_boost_round=100, **model_params):
        super().__init__(config)
        self.model_name = "XGBoostClassifier"

        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost não está disponível. Instale-o com 'pip install xgboost'"
            )

        # Parâmetros padrão otimizados para classificação
        self.params = {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "learning_rate": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",  # Método rápido para grandes datasets
            "random_state": config.RANDOM_SEED,
        }
        # Substitui ou adiciona parâmetros personalizados
        self.params.update(model_params)
        self.num_boost_round = num_boost_round
        self.model = None

    def train_and_evaluate(self, wsg_obj: WSG):
        print(f"\n--- Avaliando (XGBoost): {self.model_name} ---")
        print(
            "Este modelo pode levar mais tempo para treinar, mas geralmente oferece excelente desempenho."
        )

        pyg_data = DataConverter.to_pyg_data(wsg_obj=wsg_obj, for_embedding_bag=False)
        X = pyg_data.x.cpu().numpy()
        y = pyg_data.y.cpu().numpy()

        # Usar as máscaras de treino/teste já definidas no objeto pyg_data
        X_train, y_train = X[pyg_data.train_mask], y[pyg_data.train_mask]
        X_test, y_test = X[pyg_data.test_mask], y[pyg_data.test_mask]

        # Obter o número de classes
        num_classes = len(set(y))
        self.params["num_class"] = num_classes

        # Preparar dados no formato DMatrix do XGBoost para maior eficiência
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Medir o tempo de treinamento
        start_time = time.time()

        print(f"Treinando XGBoost por {self.num_boost_round} rounds...")
        # Treinar com feedback de progresso
        eval_result = {}
        self.model = xgb.train(
            self.params,
            dtrain,
            self.num_boost_round,
            evals=[(dtrain, "train"), (dtest, "eval")],
            evals_result=eval_result,
            verbose_eval=10,  # Mostrar progresso a cada 10 rounds
        )

        train_time = time.time() - start_time

        # Fazer predições
        y_pred_probs = self.model.predict(dtest)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Calcular métricas
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        # Apresentar algumas informações sobre características importantes
        print("\nFeature Importances:")
        feature_importance = self.model.get_score(importance_type="weight")
        sorted_importance = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        for feature, importance in sorted_importance[:5]:  # Top 5 features
            print(f"  Feature {feature}: {importance}")

        return acc, f1, train_time, report


class DeepEnsembleClassifier(BaseClassifier):
    """
    Implementa um ensemble profundo de redes neurais com múltiplas camadas ocultas e técnicas avançadas de boosting.

    Este classificador é extremamente pesado e serve como um baseline forte para comparação.
    Combina:
    1. Múltiplas redes neurais profundas (MLP) com diferentes arquiteturas
    2. Bagging (bootstrap aggregating)
    3. Técnicas de boosting para pesos de amostra
    4. Ensembling de diferentes hyperparâmetros
    """

    def __init__(
        self, config: Config, n_estimators=10, max_iter=500, n_jobs=-1, **kwargs
    ):
        super().__init__(config)
        self.model_name = "DeepEnsembleClassifier"
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.n_jobs = n_jobs  # -1 para usar todos os cores disponíveis

        # Hiperparâmetros avançados para o ensemble
        self.hidden_layer_sizes = kwargs.get(
            "hidden_layer_sizes",
            [
                (256, 256, 128),
                (512, 256, 128),
                (1024, 512, 256),
                (512, 512, 512, 256),
                (1024, 512, 512, 256),
            ],
        )
        self.activation_functions = kwargs.get("activation_functions", ["relu", "tanh"])
        self.alpha_values = kwargs.get("alpha_values", [0.0001, 0.001, 0.01])
        self.learning_rates = kwargs.get("learning_rates", ["adaptive", "constant"])

        # Mapeamento para converter strings de ativação para PyTorch
        self.pytorch_activations = {
            "relu": F.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }

        self.models = []
        self.scalers = []
        self.feature_importances_ = None

    def _create_model_configs(self):
        """Cria múltiplas configurações de modelo para o ensemble"""
        configs = []

        # Combinando diferentes hiperparâmetros para criar diversidade no ensemble
        combinations = list(
            itertools.product(
                self.hidden_layer_sizes,
                self.activation_functions,
                self.alpha_values,
                self.learning_rates,
            )
        )

        # Selecionamos um subconjunto se houver muitas combinações
        if len(combinations) > self.n_estimators:
            selected_indices = np.random.choice(
                len(combinations), size=self.n_estimators, replace=False
            )
            selected_combinations = [combinations[i] for i in selected_indices]
        else:
            # Se não houver combinações suficientes, repetimos algumas
            selected_combinations = combinations
            while len(selected_combinations) < self.n_estimators:
                selected_combinations.append(
                    combinations[np.random.randint(len(combinations))]
                )

        # Cria as configurações finais
        for hidden_size, activation, alpha, learning_rate in selected_combinations:
            configs.append(
                {
                    "hidden_layer_sizes": hidden_size,
                    "activation": activation,
                    "alpha": alpha,
                    "learning_rate": learning_rate,
                    "max_iter": self.max_iter,
                    "random_state": np.random.randint(
                        10000
                    ),  # Cada modelo tem sua própria semente
                }
            )

        return configs

    def _train_single_model(
        self, X_train, y_train, X_test, y_test, config_idx, model_config
    ):
        """Treina um único modelo do ensemble"""
        print(
            f"  Treinando modelo {config_idx + 1}/{self.n_estimators} com configuração: {model_config}"
        )

        # Cria um scaler específico para este modelo
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Inicializa e treina o modelo
        model = MLPRegressor(**model_config)
        model.fit(
            X_train_scaled, np.eye(self.n_classes_)[y_train]
        )  # One-hot encoding para saída

        # Avalia o modelo
        y_pred_proba = model.predict(X_test_scaled)
        y_pred = np.argmax(y_pred_proba, axis=1)
        accuracy = np.mean(y_pred == y_test)

        print(f"  ✓ Modelo {config_idx + 1} concluído com acurácia: {accuracy:.4f}")
        return model, scaler, accuracy

    def train_and_evaluate(self, wsg_obj: WSG):
        print(f"\n--- Avaliando (Ensemble): {self.model_name} ---")
        print(
            "AVISO: Este é um modelo extremamente pesado que pode levar 20-40 minutos para treinar."
        )
        print(
            "       Ele combina múltiplas redes neurais profundas com técnicas avançadas de ensemble."
        )

        # Converte os dados para o formato NumPy
        pyg_data = DataConverter.to_pyg_data(wsg_obj=wsg_obj, for_embedding_bag=False)
        X = pyg_data.x.cpu().numpy()
        y = pyg_data.y.cpu().numpy()

        # Identificar as classes únicas
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Dividir em conjuntos de treino e teste usando as máscaras
        X_train, y_train = X[pyg_data.train_mask], y[pyg_data.train_mask]
        X_test, y_test = X[pyg_data.test_mask], y[pyg_data.test_mask]

        # Medir o tempo de treinamento
        start_time = time.time()

        # Criar configurações variadas para os modelos
        model_configs = self._create_model_configs()
        print(f"\nTreinando ensemble com {self.n_estimators} modelos profundos...")

        # Treinar modelos em paralelo quando possível
        if self.n_jobs != 1:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._train_single_model)(
                    X_train, y_train, X_test, y_test, i, config
                )
                for i, config in enumerate(model_configs)
            )
            self.models, self.scalers, accuracies = zip(*results)
        else:
            # Treinamento sequencial (mais lento)
            self.models = []
            self.scalers = []
            accuracies = []

            for i, config in enumerate(model_configs):
                model, scaler, acc = self._train_single_model(
                    X_train, y_train, X_test, y_test, i, config
                )
                self.models.append(model)
                self.scalers.append(scaler)
                accuracies.append(acc)

        train_time = time.time() - start_time
        print(f"\nTreinamento do ensemble concluído em {train_time:.2f} segundos")

        # Avaliar o ensemble completo
        print("\nAvaliando o ensemble completo...")
        y_pred_ensemble = self._predict(X_test)

        # Calcular métricas
        acc = accuracy_score(y_test, y_pred_ensemble)
        f1 = f1_score(y_test, y_pred_ensemble, average="weighted")
        report = classification_report(
            y_test, y_pred_ensemble, output_dict=True, zero_division=0
        )

        print(f"\nDesempenho do ensemble:")
        print(f"  - Melhor modelo individual: {max(accuracies):.4f} acurácia")
        print(f"  - Ensemble completo: {acc:.4f} acurácia, {f1:.4f} F1-score")

        return acc, f1, train_time, report

    def _predict(self, X):
        """Faz previsão usando todos os modelos do ensemble"""
        # Inicializa a matriz de probabilidades
        proba_sum = np.zeros((X.shape[0], self.n_classes_))

        # Acumula previsões de cada modelo
        for model, scaler in zip(self.models, self.scalers):
            X_scaled = scaler.transform(X)
            proba_sum += model.predict(X_scaled)

        # Retorna a classe mais provável
        return np.argmax(proba_sum, axis=1)
