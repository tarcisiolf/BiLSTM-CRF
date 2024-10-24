import argparse
import os
from data_utils import load_data, to_tuples, build_vocab, save_vocab_separately, tokenize
from train_model import train
from model import save_model

def main(input_data_path, output_data_dir, output_model_dir, hyperparams):
# Carregar os dados e processar
    data_train = load_data(input_data_path)
    reports = data_train.groupby("report").apply(to_tuples).tolist()

    # Construir vocabulário e tokenizar
    word2index, index2word, tag2index, index2tag = build_vocab(data_train)

    # Verificar se o diretório de saída existe, caso contrário, criar
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)

    # Salvar vocabulário
    save_vocab_separately(word2index, index2word, tag2index, index2tag, output_data_dir)

    # Criar os dados de treinamento
    text_sequences, tag_sequences, tag_sequences_categorical = tokenize(reports, word2index, tag2index)

    # Configurações do modelo
    n_words = len(word2index)
    n_tags = len(tag2index)

    print("Dados processados e vocabulário salvo no diretório:", output_data_dir)
    print("Hiperparâmetros:")
    for key, value in hyperparams.items():
        print(f"{key}: {value}")
    
    # Aqui você pode adicionar a inicialização e treinamento do modelo
    # usando os text_sequences e tag_sequences, e aplicando os hyperparams

    model = train(hyperparams, n_words, n_tags, text_sequences, tag_sequences_categorical)
    save_model(model, output_model_dir)

if __name__ == "__main__":
    # Argumentos via linha de comando
    parser = argparse.ArgumentParser(description="Processar dados e treinar modelo BiLSTM-CRF")
    
    # Argumentos de caminho para os arquivos
    parser.add_argument('--input_data_path', type=str, required=True, help="Caminho para o arquivo de dados CSV")
    parser.add_argument('--output_data_dir', type=str, required=True, help="Diretório para salvar o vocabulário")
    parser.add_argument('--output_model_dir', type=str, required=True, help="Diretório para salvar o modelo")

    # Hiperparâmetros
    parser.add_argument('--max_len', type=int, default=512, help="Tamanho máximo da sequência de entrada")
    parser.add_argument('--lstm_dropout', type=float, default=0.1, help="Taxa de dropout na camada LSTM")
    parser.add_argument('--max_epochs', type=int, default=10, help="Número máximo de épocas de treinamento")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Taxa de aprendizado")
    parser.add_argument('--embedding_dim', type=int, default=300, help="Dimensão dos embeddings de palavras")
    parser.add_argument('--lstm_units', type=int, default=50, help="Número de unidades da LSTM")
    parser.add_argument('--batch_size', type=int, default=8, help="Tamanho do batch de treinamento")

    args = parser.parse_args()

    # Coletar hiperparâmetros em um dicionário
    hyperparams = {
        'max_len': args.max_len,
        'lstm_dropout': args.lstm_dropout,
        'max_epochs': args.max_epochs,
        'learning_rate': args.learning_rate,
        'embedding_dim': args.embedding_dim,
        'lstm_units': args.lstm_units,
        'batch_size': args.batch_size
    }

    # Executa a função principal
    main(args.input_data_path, args.output_data_dir, args.output_model_dir, hyperparams)
