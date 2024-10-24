Aqui está um exemplo de como o README para o seu repositório pode ser ajustado de acordo com a estrutura e arquivos mostrados na imagem:

---

# BiLSTM-CRF Named Entity Recognition

Este repositório contém a implementação de um modelo BiLSTM-CRF para tarefas de reconhecimento de entidade nomeada (NER). O modelo é treinado e avaliado em dados anotados com o esquema IOB.

## Estrutura do Repositório

- `data_utils.py`: Funções utilitárias para carregar e processar dados.
- `evaluate.py`: Script para avaliação do modelo com dados de teste.
- `model.py`: Definição do modelo BiLSTM-CRF e funções relacionadas.
- `requirements.txt`: Lista de dependências necessárias para executar o projeto.
- `train_model.py`: Script principal para treinamento do modelo BiLSTM-CRF.
- `train.py`: Funções auxiliares relacionadas ao treinamento.
- `utils.py`: Funções adicionais, como a criação de dataframes com os resultados.

## Dependências

Para instalar as dependências necessárias, execute o seguinte comando:

```bash
pip install -r requirements.txt
```

## Treinamento do Modelo

Para treinar o modelo, use o arquivo `train_model.py`. Você pode especificar os dados de treinamento e vocabulário, além dos hiperparâmetros diretamente pela linha de comando. Exemplo:

```bash
python train_model.py --data_path data/df_train.csv --output_dir data/vocab --max_len 512 --lstm_dropout 0.1 --max_epochs 10 --learning_rate 0.01 --embedding_dim 300 --lstm_units 50 --batch_size 8
```

### Parâmetros:
- `--data_path`: Caminho para o arquivo de dados CSV de treinamento.
- `--output_dir`: Diretório para salvar os arquivos de vocabulário.
- `--max_len`: Tamanho máximo das sequências.
- `--lstm_dropout`: Taxa de dropout na LSTM.
- `--max_epochs`: Número de épocas de treinamento.
- `--learning_rate`: Taxa de aprendizado.
- `--embedding_dim`: Dimensão dos embeddings de palavras.
- `--lstm_units`: Número de unidades na LSTM.
- `--batch_size`: Tamanho do batch de treinamento.

## Avaliação do Modelo

O arquivo `evaluate.py` permite a avaliação do modelo treinado em um conjunto de dados de teste. Ele carrega o modelo salvo e os arquivos de vocabulário e realiza previsões. Exemplo:

```bash
python evaluate.py --data_path data/df_test.csv --vocab_dir data --model_path models/model_00 --max_sentence 512
```

### Parâmetros:
- `--data_path`: Caminho para o arquivo CSV com os dados de teste.
- `--vocab_dir`: Diretório contendo os arquivos de vocabulário (`index2tag.json`, `index2word.json`, etc.).
- `--model_path`: Caminho para o modelo salvo.
- `--max_sentence`: Tamanho máximo das sequências.

## Estrutura de Dados

Os dados de entrada devem estar no formato CSV e devem conter as colunas correspondentes às tokens e suas respectivas labels IOB. O vocabulário gerado durante o treinamento é salvo no diretório especificado pelo usuário.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request.

## Licença

Este projeto está licenciado sob os termos da licença MIT.

---

Este README agora inclui instruções detalhadas sobre como executar o treinamento e a avaliação do modelo, além de descrever a estrutura do repositório.
