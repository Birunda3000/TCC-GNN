# TCC-GNN: Framework para Geração e Análise de Embeddings de Grafos

Este repositório contém o código-fonte de um framework desenvolvido para experimentação com Redes Neurais de Grafos (GNNs). O foco principal é a geração de *node embeddings* (representações vetoriais de nós) de forma auto-supervisionada usando um *Variational Graph Autoencoder* (VGAE) e a subsequente avaliação da qualidade desses embeddings em tarefas de classificação de nós.

O projeto é construído com ênfase em modularidade, reprodutibilidade e um pipeline de dados bem definido, utilizando um formato de dados customizado chamado **Weighted Sparse Graph (WSG)**.

## Principais Funcionalidades

-   **Formato de Dados Padronizado (WSG):** Define uma especificação (`.wsg.json`) para representar grafos de maneira universal, desacoplando a preparação dos dados da modelagem.
-   **Ambiente Reproduzível com Docker:** Configuração completa para `dev containers` do VS Code, com suporte para ambientes **CPU** e **GPU (NVIDIA)**, garantindo que o projeto funcione de forma consistente em diferentes máquinas.
-   **Pipeline Modular:**
    1.  **Carregamento de Dados:** Converte datasets brutos (ex: Musae-Github) para o formato WSG.
    2.  **Geração de Embeddings:** Treina um modelo VGAE para aprender representações de nós e salva os embeddings resultantes em um novo arquivo WSG.
    3.  **Avaliação em Tarefas Downstream:** Utiliza os embeddings gerados para treinar e avaliar modelos de classificação (MLP, Regressão Logística, etc.).
-   **Geração de Embeddings com VGAE:** Implementação de um *Variational Graph Autoencoder* em PyTorch Geometric para aprender embeddings de nós de forma auto-supervisionada.
-   **Anonimização de Features:** Inclui um script para aplicar privacidade diferencial (mecanismo de Laplace) às features dos nós e avaliar o impacto na utilidade dos dados para tarefas de classificação.
-   **Gerenciamento de Experimentos:** Salva automaticamente os resultados de cada execução (modelo treinado, embeddings, logs e métricas) em diretórios nomeados de forma descritiva.

## Estrutura do Projeto

```
/app/gnn_tcc/
├── .devcontainer/         # Configurações do Docker e VS Code Dev Container (CPU/GPU)
├── data/
│   ├── datasets/          # Datasets brutos (ex: musae-github)
│   └── output/            # Diretório para salvar os resultados dos experimentos
├── src/                   # Código-fonte principal do projeto
│   ├── __init__.py
│   ├── anon.py            # Script para anonimização de dados
│   ├── classifiers.py     # Definição de modelos de classificação (MLP, GCN)
│   ├── config.py          # Configurações centralizadas do projeto
│   ├── data_converter.py  # Conversor do formato WSG para PyTorch Geometric
│   ├── data_format_definition.py # Definição Pydantic do formato WSG
│   ├── data_loader.py     # Loaders para carregar datasets para o formato WSG
│   ├── directory_manager.py # Gerenciador de diretórios de saída
│   ├── model.py           # Definição do modelo VGAE
│   ├── train.py           # Lógica de treinamento do VGAE
│   └── train_loop.py      # Lógica de treinamento para os classificadores
├── requirements.txt       # Dependências Python
├── run_embedding_generation.py # Script principal para gerar embeddings
└── run_feature_classification.py # Script principal para avaliar classificadores
```

## Como Começar

Este projeto foi projetado para ser executado dentro de um ambiente de desenvolvimento em contêiner, o que simplifica a configuração.

### Pré-requisitos

-   [Docker](https://www.docker.com/get-started)
-   [Visual Studio Code](https://code.visualstudio.com/)
-   Extensão [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) para o VS Code.

### Configuração do Ambiente

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd gnn_tcc
    ```

2.  **Escolha o ambiente (CPU ou GPU):**
    -   Abra o arquivo `.devcontainer/devcontainer.json`.
    -   Por padrão, ele está configurado para usar o ambiente de CPU:
        ```json
        "dockerComposeFile": "docker-compose.cpu.yml",
        ```
    -   Se você possui uma GPU NVIDIA e os drivers corretos instalados, pode mudar para o ambiente de GPU:
        ```json
        // "dockerComposeFile": "docker-compose.cpu.yml",
        "dockerComposeFile": "docker-compose.gpu.yml",
        ```

3.  **Abra o projeto no Dev Container:**
    -   Abra o VS Code.
    -   Pressione `F1` para abrir a paleta de comandos e selecione **"Dev Containers: Reopen in Container"**.
    -   O VS Code irá construir a imagem Docker e iniciar o contêiner. Este processo pode levar alguns minutos na primeira vez.

## Executando o Pipeline

O fluxo de trabalho principal é dividido em dois scripts.

### 1. Gerar os Embeddings

Este script carrega um dataset bruto, treina o modelo VGAE e salva os embeddings resultantes em um novo arquivo `.wsg.json`.

1.  **Configure o dataset:**
    -   Abra `src/config.py` e ajuste a variável `DATASET_NAME` se necessário.

2.  **Execute o script:**
    -   Abra um terminal no VS Code (que já estará dentro do contêiner) e execute:
    ```bash
    python run_embedding_generation.py
    ```
    -   Ao final da execução, o caminho para o diretório de resultados será exibido no terminal. Este diretório conterá o modelo treinado (`vgae_model.pt`), um resumo da execução (`run_summary.txt`) e o arquivo com os embeddings (`<dataset>_embeddings.wsg.json`).

### 2. Avaliar os Embeddings em Classificação

Este script carrega um arquivo `.wsg.json` (que pode conter os embeddings gerados na etapa anterior) e avalia o desempenho de diferentes classificadores na tarefa de classificação de nós.

1.  **Configure o arquivo de entrada:**
    -   Abra `run_feature_classification.py`.
    -   Atualize a variável `wsg_file_path` com o caminho para o arquivo `.wsg.json` que você deseja avaliar (gerado na etapa anterior).

2.  **Execute o script:**
    ```bash
    python run_feature_classification.py
    ```
    -   O script irá treinar e avaliar modelos como Regressão Logística, Random Forest e um MLP, exibindo um relatório comparativo de acurácia, F1-Score e tempo de treino.
