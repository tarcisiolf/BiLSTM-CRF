import argparse
import tensorflow as tf
from model import CRF, load_model
from data_utils import load_data, to_tuples, tokenize, load_vocab
from utils import process_test_sentences, create_dataframe_with_results


def main(data_path, vocab_dir, model_path, max_sentence):
    # Carregar os dados e processar
    data_test = load_data(data_path)
    index2tag = load_vocab(f'{vocab_dir}/index2tag.json')
    tag2index = load_vocab(f'{vocab_dir}/tag2index.json')
    index2word = load_vocab(f'{vocab_dir}/index2word.json')
    word2index = load_vocab(f'{vocab_dir}/word2index.json')

    test_reports = data_test.groupby("report").apply(to_tuples).tolist()

    # Tokenizar o vocabulário
    test_text_sequences, test_tag_sequences, test_tag_sequences_categorical = tokenize(test_reports, word2index, tag2index)

    # Processa as sentenças de teste, convertendo índices em palavras e tags.
    test_sentences, test_tags = process_test_sentences(index2tag, index2word, test_text_sequences, test_tag_sequences_categorical)

    # Carregar o modelo
    loaded_model = load_model(model_path)

    # Criar e salvar o Dataframe com os resultados da previsão do modelo
    create_dataframe_with_results(test_sentences, test_tags, loaded_model, word2index, index2tag, MAX_SENTENCE=max_sentence)

    print(f"Resultados salvos usando o modelo: {model_path}")


if __name__ == '__main__':
    # Argumentos via linha de comando
    parser = argparse.ArgumentParser(description="Avaliar modelo BiLSTM-CRF")
    
    # Argumentos para arquivos e diretórios
    parser.add_argument('--data_path', type=str, required=True, help="Caminho para o arquivo de dados CSV de teste")
    parser.add_argument('--vocab_dir', type=str, required=True, help="Diretório contendo os arquivos de vocabulário (index2tag, tag2index, index2word, word2index)")
    parser.add_argument('--model_path', type=str, required=True, help="Caminho para o modelo treinado")
    parser.add_argument('--max_sentence', type=int, default=512, help="Tamanho máximo da sentença")

    args = parser.parse_args()

    # Executa a função principal
    main(args.data_path, args.vocab_dir, args.model_path, args.max_sentence)




