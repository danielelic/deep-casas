#!/usr/bin/env python3

import argparse
import csv
from datetime import datetime

import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight

import data
import models

# fix random seed for reproducibility
seed = 7
units = 64
epochs = 200

if __name__ == '__main__':
    """The entry point"""
    # set and parse the arguments list
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='')
    p.add_argument('--v', dest='model', action='store', default='', help='deep model')
    args = p.parse_args()

    print(data.datasetsNames)
    for dataset in data.datasetsNames:
        X, Y, dictActivities = data.getData(dataset)

        cvaccuracy = []
        cvscores = []
        modelname = ''

        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        k = 0
        for train, test in kfold.split(X, Y):
            print('X_train shape:', X[train].shape)
            print('y_train shape:', Y[train].shape)

            print(dictActivities)
            args_model = str(args.model)

            if 'Ensemble' in args_model:
                input_dim = len([X[train], X[train]])
                X_train_input = [X[train], X[train]]
                X_test_input = [X[test], X[test]]
            else:
                input_dim = len(X[train])
                X_train_input = X[train]
                X_test_input = X[test]
            no_activities = len(dictActivities)

            if args_model == 'LSTM':
                model = models.get_LSTM(input_dim, units, data.max_lenght, no_activities)
            elif args_model == 'biLSTM':
                model = models.get_biLSTM(input_dim, units, data.max_lenght, no_activities)
            elif args_model == 'Ensemble2LSTM':
                model = models.get_Ensemble2LSTM(input_dim, units, data.max_lenght, no_activities)
            elif args_model == 'CascadeEnsembleLSTM':
                model = models.get_CascadeEnsembleLSTM(input_dim, units, data.max_lenght, no_activities)
            elif args_model == 'CascadeLSTM':
                model = models.get_CascadeLSTM(input_dim, units, data.max_lenght, no_activities)
            else:
                print('Please get the model name '
                      '(eg. --v [LSTM | biLSTM | Ensemble2LSTM | CascadeEnsembleLSTM | CascadeLSTM])')
                exit(-1)

            model = models.compileModel(model)
            modelname = model.name

            currenttime = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
            csv_logger = CSVLogger(
                model.name + '-' + dataset + '-' + str(currenttime) + '.csv')
            model_checkpoint = ModelCheckpoint(
                model.name + '-' + dataset + '-' + str(currenttime) + '.h5',
                monitor='acc',
                save_best_only=True)

            # train the model
            print('Begin training ...')
            class_weight = compute_class_weight('balanced', np.unique(Y),
                                                Y)  # use as optional argument in the fit function

            model.fit(X_train_input, Y[train], validation_split=0.2, epochs=epochs, batch_size=64, verbose=1,
                      callbacks=[csv_logger, model_checkpoint])

            # evaluate the model
            print('Begin testing ...')
            scores = model.evaluate(X_test_input, Y[test], batch_size=64, verbose=1)
            print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

            print('Report:')
            target_names = sorted(dictActivities, key=dictActivities.get)

            classes = model.predict_classes(X_test_input, batch_size=64)
            print(classification_report(list(Y[test]), classes, target_names=target_names))
            print('Confusion matrix:')
            labels = list(dictActivities.values())
            print(confusion_matrix(list(Y[test]), classes, labels))

            cvaccuracy.append(scores[1] * 100)
            cvscores.append(scores)

            k += 1

        print('{:.2f}% (+/- {:.2f}%)'.format(np.mean(cvaccuracy), np.std(cvaccuracy)))

        currenttime = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        csvfile = 'cv-scores-' + modelname + '-' + dataset + '-' + str(currenttime) + '.csv'

        with open(csvfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in cvscores:
                writer.writerow([",".join(str(el) for el in val)])
