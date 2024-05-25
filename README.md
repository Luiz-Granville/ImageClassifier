# Classificação de Imagens com CIFAR-10

## Descrição

Este projeto implementa um modelo de rede neural convolucional (CNN) para classificar imagens do conjunto de dados CIFAR-10. O modelo é exposto via uma API que recebe uma imagem e retorna a categoria prevista.

## Estrutura do Projeto

```plaintext
/project-root
├── /api
│ └── app.py # Código da API 
├── /images # Imagens para predizer
│ .
│ .
│ .
├── /model
│ ├── train.py # Script para treinamento do modelo
│ ├── model.py # Definição da arquitetura do modelo
│ ├── utils.py # Funções utilitárias para carregar dados
│ └── model_weights.h5 # Arquivo com os pesos do modelo treinado
├── test.py # Script de teste da API
├── README.md # Instruções de setup e uso
├── report.md # Relatório detalhado do projeto
└── requirements.txt # Dependências da API
```

## Configuração e Uso

1. Instale as dependências necessárias:
    ```sh
    pip install -r /requirements.txt
    ```

### Treinamento do Modelo

1. Execute o script de treinamento:
    ```sh
    python model/train.py
    ```

### Executando a API

1. Execute o aplicativo Flask:
    ```sh
    python api/app.py
    ```

2. A API estará disponível em `http://127.0.0.1:5000/predict`. Envie uma imagem via POST request para obter a categoria prevista.

### Exemplo de Uso da API

Envie uma imagem para a API usando o seguinte comando curl:
```sh
curl -X POST -F "image=@path_to_image.jpg" http://127.0.0.1:5000/predict
```

Substitua path_to_image.jpg pelo caminho para a sua imagem.

#### ou

### Executando test.py

1. Execute o aplicativo Flask:
    ```sh
    python test.py
    ```