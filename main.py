"""
        ACA Project

        ATHORS:
            Joana Simões
            Pedro Carrasco
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import sys
import datetime
import json
import pprint
import time
from tensorflow.keras import layers, models
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
# to ignore terminal warnings
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import random
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input

# plot parameters
plt.rcParams["figure.figsize"] = (20,12)
plt.rcParams['axes.grid'] = True
plt.style.use('fivethirtyeight')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['lines.linewidth'] = 3


# fix seeds: code from https://stackoverflow.com/questions/36288235/how-to-get-stable-results-with-tensorflow-setting-random-seed
def set_global_determinism(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# Call the above function with seed value


def get_layer(layer_config):
    layer_config = layer_config.copy()
    print(layer_config)
    layer_name = layer_config.pop('type')
    layer_func = getattr(tf.keras.layers, layer_name)
    layer = layer_func(**layer_config)
    return layer

def get_model(config, num_classes, width=50, height=50):
    loss = getattr(tf.keras.losses, config['loss'])
    optimizer = getattr(tf.keras.optimizers, config['optimizer'])
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(width, height, 3)))
    for layer in config["layers"]:
        model.add(get_layer(layer))
    model.add(layers.Dense(units=num_classes, activation='softmax'))

    # base_model = Xception(weights='imagenet', include_top=False)
    # base_model.trainable = False
    # loss = getattr(tf.keras.losses, config['loss'])
    # model = models.Sequential(
    #     [   
    #         layers.InputLayer(input_shape=(width, height, 3)),
    #         layers.Resizing(width=72, height=72),
    #         base_model,
    #         layers.Dropout(0.2),
    #         layers.Flatten(),
    #         layers.Dense(50, activation='relu'),
    #         layers.Dense(20, activation='relu'),
    #         layers.Dense(units=num_classes, activation='softmax')
    #      ]
    # )
    model.compile(
        optimizer=optimizer(learning_rate=config['learning_rate']),
        loss=loss(),
        metrics=config['metrics'],
    )
    return model

def write_predictions(preds):
    test_df = pd.DataFrame()
    test_df['number'] = np.arange(len(preds))
    test_df['class'] = preds
    return test_df

def fit_and_predict_dataset(config, num_classes, x_train, x_test, y_train, y_test=None, prefix='', path='', seed=42):
    # with train dataset and test split
    model = get_model(config, num_classes)
    # early stopping
    early_stopping_loss = EarlyStopping(
        monitor = 'val_loss', 
        patience = config['patience'], 
        restore_best_weights = True,
        )
    early_stopping_accuracy = EarlyStopping(monitor = 'val_accuracy', patience = config['patience'], mode='max', restore_best_weights=True)
    # split train data into train and val
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=config['validation_split'], shuffle=True, stratify=y_train, random_state=seed)
    # fit model
    history = model.fit(
        x_train, 
        y_train, 
        # validation_split=config['validation_split'], 
        validation_data = (x_val, y_val),
        epochs=config['epochs'],
        callbacks=[early_stopping_loss],
        batch_size=config['batch_size']
    )
    model.save(os.path.join(path, f'{prefix}_model.h5'))
    with open(os.path.join(LOGS_PATH, f'{prefix}_model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # predict test from train dataset
    predictions = model.predict(x_test)
    preds = np.argmax(predictions, axis=1)

    test_df = write_predictions(preds)


    pprint.pprint(history.history)
    
    if y_test is not None:
        print(metrics.classification_report(y_test, preds))
        with open(os.path.join(path, f'{prefix}_classification_report.txt'), 'w') as file:
            file.write(metrics.classification_report(y_test, preds))
        test_df['real'] = y_test
        # save predictions
        test_df.to_csv(os.path.join(path, f'{prefix}_prediction_{exp}.csv'), index=False)
    else:
        test_df.to_csv(os.path.join(path, f'{prefix}_prediction_{exp}.csv'), index=False)
    with open(os.path.join(path, f'{prefix}_history.json'), 'w') as outfile:
        json.dump(history.history, outfile, indent=1)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(LOGS_PATH, f'{prefix}_train_results.png'))
    # plt.show()
    return preds

def ensemble_models(config, num_classes, x_train, x_test, y_train, y_test, prefix='ensemble_train', path='', seed=42):
    predictions = pd.DataFrame()
    indexes = np.arange(0, len(x_train))
    for i in range(config['ensemble_n_estimators']):
        # with replacement
        sample_indexes = np.random.choice(indexes, int(len(x_train*0.7)))
        x_train_aux = x_train[sample_indexes]
        y_train_aux = y_train[sample_indexes]
        preds = fit_and_predict_dataset(config.copy(), num_classes, x_train_aux, x_test, y_train_aux, y_test=y_test, prefix=f'{prefix}_{i}',path=path, seed=seed)
        predictions[f'model_{i}'] = preds

    p = predictions.mode(axis=1).to_numpy()[:, 0].astype(int)
    return p

# main funtion of python
if __name__ == '__main__':
    seed = 1257623
    set_global_determinism(seed)
    # get experience from logs folder
    experiences = os.listdir(os.path.join('logs'))
    # if logs folder is empty
    if len(experiences) == 0:
        exp = 'exp00'
    else:
        # in case the logs files doesn´t contain the name "exp"
        try:
            num = int(sorted(experiences)[-1][3:]) + 1
            # in case num smaller than 10 to register the files in a scale of 00 to 09
            if num < 10:
                exp = f'exp0{num}'
            # otherwise
            else:
                exp = f'exp{num}'
        except:
            exp = f'exp{datetime.datetime.now()}'

    # crate the new experience folder
    LOGS_PATH = os.path.join('logs', exp)
    os.makedirs(LOGS_PATH)
    # PREDICTIONS_PATH = os.path.join(LOGS_PATH, 'predictions')
    # os.makedirs(PREDICTIONS_PATH)
    # TARGET_PATH = os.path.join(LOGS_PATH, 'kaggle')
    # os.makedirs(TARGET_PATH)

    # gets the name of the config file and reads it
    config_path = sys.argv[1]
    with open(config_path, 'r') as file:
        config = json.load(file)
        
    # save config in logs folder
    with open(os.path.join(LOGS_PATH, 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=1)

    path = os.path.join('dataset')
    X = np.load(os.path.join(path, 'trainX.npy'))
    y = np.load(os.path.join(path, 'trainy.npy'), allow_pickle=True).astype(int)
    X_test = np.load(os.path.join(path, 'testX.npy'))

    X = X / 255
    X_test = X_test / 255

    width = 50
    height = 50
    num_classes = 6

    if config.get('cross-validation', False):
        accuracies = []
        kfold = StratifiedKFold(shuffle=True, random_state=seed)
        for fold, (train_indexes, test_indexes) in enumerate(kfold.split(X, y)):
            x_train = X[train_indexes]
            x_test = X[test_indexes]
            y_train = y[train_indexes]
            y_test = y[test_indexes]
            # fit and predict train dataset
            print(f'FOLD {fold} - PREDICT TRAIN DATASET')
            preds = fit_and_predict_dataset(config.copy(), num_classes, x_train, x_test, y_train, y_test=y_test, prefix=f'test_split_fold{fold}', path=LOGS_PATH, seed=seed)
            accuracies.append(metrics.accuracy_score(y_test, preds))
        
        print(accuracies)
        print('AVERAGE ACCURACY', np.mean(accuracies))
    elif config.get('ensemble', False):
        print('Ensemble models')
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=config['train_split'], shuffle=True, stratify=y, random_state=seed )
        preds = ensemble_models(config.copy(), num_classes, x_train, x_test, y_train, y_test, prefix='ensemble_train', path=LOGS_PATH, seed=seed)
        with open(os.path.join(path, f'ensemble_classification_report.txt'), 'w') as file:
            file.write(metrics.classification_report(y_test, preds))
        print(metrics.classification_report(y_test, preds))
    else:
        print('PREDICT TRAIN DATASET')
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=config['train_split'], shuffle=True, stratify=y, random_state=seed )
        preds = fit_and_predict_dataset(config.copy(), num_classes, x_train, x_test, y_train, y_test=y_test, prefix=f'test_split',path=LOGS_PATH, seed=seed)

    if config.get('predict-kaggle', 'False'):
        # predict kaggle dataset
        print('PREDICT KAGGLE DATASET')
        # print(config)
        fit_and_predict_dataset(config.copy(), num_classes, X, X_test, y, y_test=None, prefix='kaggle', path=LOGS_PATH, seed=seed)
        
        if config.get('ensemble', False):
            print('Ensemble models')
            # x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=config['train_split'], shuffle=True, stratify=y, random_state=seed)
            preds = ensemble_models(config.copy(), num_classes, X, X_test, y, y_test=None, prefix='kaggle_ensemble', path=LOGS_PATH, seed=seed)
            test_df = write_predictions(preds)
            test_df.to_csv(os.path.join(LOGS_PATH, f'{exp}_kaggle_ensemble_all.csv'), index=False)