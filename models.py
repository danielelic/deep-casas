#!/usr/bin/env python3

from keras.layers import Dense, LSTM, Bidirectional, Merge
from keras.layers.embeddings import Embedding
from keras.models import Sequential


def get_LSTM(input_dim, output_dim, max_lenght, no_activities):
    model = Sequential(name='LSTM')
    model.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model.add(LSTM(output_dim))
    model.add(Dense(no_activities, activation='softmax'))
    return model


def get_biLSTM(input_dim, output_dim, max_lenght, no_activities):
    model = Sequential(name='biLSTM')
    model.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model.add(Bidirectional(LSTM(output_dim)))
    model.add(Dense(no_activities, activation='softmax'))
    return model


def get_Ensemble2LSTM(input_dim, output_dim, max_lenght, no_activities):
    model1 = Sequential()
    model1.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model1.add(Bidirectional(LSTM(output_dim)))

    model2 = Sequential()
    model2.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model2.add(LSTM(output_dim))

    model = Sequential(name='Ensemble2LSTM')
    model.add(Merge([model1, model2], mode='concat'))
    model.add(Dense(no_activities, activation='softmax'))
    return model


def get_CascadeEnsembleLSTM(input_dim, output_dim, max_lenght, no_activities):
    model1 = Sequential()
    model1.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model1.add(Bidirectional(LSTM(output_dim, return_sequences=True)))

    model2 = Sequential()
    model2.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model2.add(LSTM(output_dim, return_sequences=True))

    model = Sequential(name='CascadeEnsembleLSTM')
    model.add(Merge([model1, model2], mode='concat'))
    model.add(LSTM(output_dim))
    model.add(Dense(no_activities, activation='softmax'))
    return model


def get_CascadeLSTM(input_dim, output_dim, max_lenght, no_activities):
    model = Sequential(name='CascadeLSTM')
    model.add(Embedding(input_dim, output_dim, input_length=max_lenght, mask_zero=True))
    model.add(Bidirectional(LSTM(output_dim, return_sequences=True)))
    model.add(LSTM(output_dim))
    model.add(Dense(no_activities, activation='softmax'))
    return model


def compileModel(model):
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
