import numpy as np
import pandas as pd
import operator
import sys
import re
from string import punctuation

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

sys.path.append('..')
from model.text_cnn import TextCNN
from data_utils import utils


def create_glove_embeddings(embed_file, word_index, max_num_words):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embed_file))
    embedding_dim = 300
    embedding_matrix = np.zeros((max_num_words, embedding_dim))

    for word, i in word_index.items():
        if i >= max_num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def experiment_with_imdb():

    train_texts, train_label, test_texts, test_label = utils.load_imdb()

    config = {
        'MAX_NUM_WORDS': 15000,
        'MAX_TEXT_LEN': 500,
        'NUM_CLASSES': 2,
        'FILTER_SIZES': [2, 3, 4, 5],
        'FILTER_NUM': 200,
        'EMBED_DROPOUT': 0.3,
        'DENSE_DROPOUT': 0.5,

        'BATCH_SIZE': 64,
        'EPOCHS': 10,
    }

    tokenizer = Tokenizer(num_words=config['MAX_NUM_WORDS'])
    tokenizer.fit_on_texts(train_texts)

    train_texts = tokenizer.texts_to_sequences(train_texts)
    test_texts = tokenizer.texts_to_sequences(test_texts)

    x_train = pad_sequences(train_texts, maxlen=config['MAX_TEXT_LEN'])
    x_test = pad_sequences(test_texts, maxlen=config['MAX_TEXT_LEN'])

    x_train, x_val, y_train, y_val = train_test_split(x_train, train_label, train_size=0.8, random_state=2018)

    matrix = create_glove_embeddings(embed_file='../datasets/glove.840B.300d.txt', word_index=tokenizer.word_index,
                                     max_num_words=config['MAX_NUM_WORDS'])

    model = TextCNN(matrix, maxlen=config['MAX_TEXT_LEN'], num_classes=config['NUM_CLASSES'],
                    filter_sizes=config['FILTER_SIZES'], filter_num=config['FILTER_NUM'],
                    embed_dropout=config['EMBED_DROPOUT'], dense_dropout=config['DENSE_DROPOUT'])

    model.fit(x=x_train, y=y_train, epochs=config['EPOCHS'], batch_size=config['BATCH_SIZE'],
              validation_data=(x_val, y_val), save_model=True)

    model.load_weight('../tmp/text_cnn')

    test_pred = model.predict(x_test)
    from sklearn.metrics import accuracy_score
    print('acc on test data: {}'.format(accuracy_score(test_label, test_pred)))


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
experiment_with_imdb()
