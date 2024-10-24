import pandas as pd
import numpy as np

"""
def number_to_word_test_sentences_and_tags(index2tag, index2word, X_test, y_test):

    test_sentences= []
    test_tags = []

    # Recupera os laudos e tags no formato word2index/tag2index
    for i in range(len(X_test)):
        aux_tag = []

        report = ""
        sentence = X_test[i]
        tags = y_test[i]

        # Recupera o laudo
        for j in range(len(sentence)):
            # Recupera a palavra
            word = sentence[j]
            # Recupera a tag
            tag = tags[j]
            
            #print(word)
            #print(tag)
            
            # A tag é predita em one-hot-enconding
            # int_tag é o inteiro que representa a tag
            # no dicionário index2tag
            int_tag = np.where(tag == int(1))
            
            #print(int_tag)
            
            # Constrói o laudo ignorando as palavras "padding"
            # Constrói o array de tags do laudo
            if str(index2word[word]) != '--PADDING--':
                report = report + " " + str(index2word[word])
                aux_tag.append(index2tag[int(int_tag[0][0])])

        
        #print(report)
        #print(aux_tag)
        
        test_sentences.append(report)
        test_tags.append(aux_tag)

    return test_sentences, test_tags

    def train_test_df_model(train_sentences, train_tags):

    train_df = pd.DataFrame(columns = ['report', 'word', 'tag'])

    for i in range (len(train_sentences)):

        # Gera os laudos no formato index2word com o tamanho max_sentence

        #print("LAUDO " + str(i) + "____________________________________________________________________________________________")

        sentence = train_sentences[i]
        tags = train_tags[i]
        
        sentence = sentence.split()
        
        #print(len(padded_sentence))
        #print(len(tags))
        #print(len(pred[0]))
        #print(pred[0])

        if i < 10:
            retval = ""
            for w, t in zip(sentence, tags):
                retval = retval + "{:25}: {:10}".format(w, t) + "\n"
                aux_dict = {'report': ('report_0' + str(i)), 'word': w, 'tag' : t}
                df_new_row = pd.DataFrame([aux_dict])
                train_df = pd.concat([train_df, df_new_row])
                #test_df = test_df.append({'sentence': ('sentence_0' + str(i)), 'word': w, 'tag' : t, 'tag_pred' : index2tag[p]}, ignore_index = True)


        else:
            retval = ""
            for w, t in zip(sentence, tags):
                retval = retval + "{:25}: {:10}".format(w, t) + "\n"
                aux_dict = {'report': ('report_' + str(i)), 'word': w, 'tag' : t}
                df_new_row = pd.DataFrame([aux_dict])
                train_df = pd.concat([train_df, df_new_row])
                #test_df = test_df.append({'sentence': ('sentence_0' + str(i)), 'word': w, 'tag' : t, 'tag_pred' : index2tag[p]}, ignore_index = True)

        #print(retval)

    return train_df

def result_df_model_previous(test_sentences, test_tags, model, word2index, index2tag, MAX_SENTENCE):

    test_df = pd.DataFrame(columns = ['report', 'word', 'tag', 'tag_pred'])

    for i in range (len(test_sentences)):

        # Gera os laudos no formato index2word com o tamanho max_sentence

        #print("LAUDO " + str(i) + "____________________________________________________________________________________________")
        sentence = test_sentences[i]
        tags = test_tags[i]
        
        sentence = sentence.split()
        padded_sentence = sentence + [word2index["--PADDING--"]] * (MAX_SENTENCE - len(sentence))
        padded_sentence = [word2index.get(w, 0) for w in padded_sentence]

        # Faz a predição das tags das palavras
        pred = model.predict(np.array([padded_sentence]))
        pred = np.argmax(pred, axis=-1)

        #print(len(padded_sentence))
        #print(len(tags))
        #print(len(pred[0]))
        #print(pred[0])

        if i < 10:
            retval = ""
            for w, t, p in zip(sentence, tags, pred[0]):
                retval = retval + "{:25}: {:10} {:5}".format(w, t, index2tag[p]) + "\n"
                aux_dict = {'report': ('report_0' + str(i)), 'word': w, 'tag' : t, 'tag_pred' : index2tag[p]}
                df_new_row = pd.DataFrame([aux_dict])
                test_df = pd.concat([test_df, df_new_row])
                #test_df = test_df.append({'sentence': ('sentence_0' + str(i)), 'word': w, 'tag' : t, 'tag_pred' : index2tag[p]}, ignore_index = True)


        else:
            retval = ""
            for w, t, p in zip(sentence, tags, pred[0]):
                retval = retval + "{:25}: {:10} {:5}".format(w, t, index2tag[p]) + "\n"
                aux_dict = {'report': ('report_' + str(i)), 'word': w, 'tag' : t, 'tag_pred' : index2tag[p]}
                df_new_row = pd.DataFrame([aux_dict])
                test_df = pd.concat([test_df, df_new_row])
                #test_df = test_df.append({'sentence': ('sentence_0' + str(i)), 'word': w, 'tag' : t, 'tag_pred' : index2tag[p]}, ignore_index = True)

        #print(retval)

    return test_df
"""

def process_test_sentences(index2tag, index2word, X_test, y_test):
    """
    Processa as sentenças de teste, convertendo índices em palavras e tags.

    Args:
        index2tag (dict): Dicionário que mapeia índices para tags.
        index2word (dict): Dicionário que mapeia índices para palavras.
        X_test (list): Lista de sentenças de teste, onde cada sentença é uma lista de índices.
        y_test (list): Lista de tags correspondentes às sentenças de teste.

    Returns:
        tuple: Uma tupla contendo duas listas: 
            - test_sentences (list): Lista de sentenças de teste em texto.
            - test_tags (list): Lista de tags correspondentes às sentenças de teste.

    """
    test_sentences = []
    test_tags = []

    # Itera sobre as sentenças de teste e suas respectivas tags
    for sentence, tags in zip(X_test, y_test):
        report = []
        aux_tag = []

        # Itera sobre as palavras da sentença e suas tags
        for word, tag in zip(sentence, tags):
            # Verifica se a palavra não é padding
            
            word = str(word)
            if index2word[word] != '--PADDING--':
                # Adiciona a palavra ao laudo e a tag correspondente
                report.append(index2word[word])
                int_tag = np.argmax(tag)  # Obtém o índice da tag
                aux_tag.append(index2tag[str(int_tag)])

        # Adiciona o laudo e as tags à lista de sentenças e tags
        test_sentences.append(" ".join(report))
        test_tags.append(aux_tag)

    return test_sentences, test_tags

def create_dataframe_with_results(test_sentences, test_tags, model, word2index, index2tag, MAX_SENTENCE):
    """
    Cria um DataFrame com os resultados da predição do modelo para as sentenças de teste.

    Args:
        test_sentences (list): Lista de sentenças de teste em texto.
        test_tags (list): Lista de tags correspondentes às sentenças de teste.
        model: Modelo de predição de tags.
        word2index (dict): Dicionário que mapeia palavras para índices.
        index2tag (dict): Dicionário que mapeia índices para tags.
        MAX_SENTENCE (int): Tamanho máximo de uma sentença.

    Returns:
        pd.DataFrame: DataFrame com os resultados da predição, contendo as colunas:
            - report: Identificador do relatório.
            - word: Palavra da sentença.
            - tag: Tag real da palavra.
            - tag_pred: Tag predita pela modelo.

    """
    test_df = pd.DataFrame(columns=['report', 'word', 'tag', 'tag_pred'])
    
    for i, (sentence, tags) in enumerate(zip(test_sentences, test_tags)):
        #print("report {}", i)
        # Prepara a sentença com padding
        sentence_words = sentence.split()
        padded_sentence = sentence_words + ['--PADDING--'] * (MAX_SENTENCE - len(sentence_words))
        padded_sentence = [word2index.get(word, 0) for word in padded_sentence]

        # Faz a predição das tags
        pred = model.predict(np.array([padded_sentence]))
        pred = np.argmax(pred, axis=-1)[0]

        report_id = f'report_{"0" if i < 10 else ""}{i}'

        # Itera sobre as palavras, tags reais e tags preditas
        for word, true_tag, pred_tag in zip(sentence_words, tags, pred):
            aux_dict = {
                'report': report_id,
                'word': word,
                'tag': true_tag,
                'tag_pred': index2tag[str(pred_tag)]
            }
            test_df = pd.concat([test_df, pd.DataFrame([aux_dict])])
    
    test_df.to_csv("data/dataframe_with_results_of_model.csv", encoding='utf-8', index=False)
    return

