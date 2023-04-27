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
from sklearn.model_selection import StratifiedGroupKFold
# to ignore terminal warnings
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import random

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
    layer_name = layer_config.pop('type')
    layer_func = getattr(tf.keras.layers, layer_name)
    layer = layer_func(**layer_config)
    return layer

def get_model(config, num_classes, width=50, height=50):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(width, height, 3,)))
    
    for layer in config["layers"]:
        print(layer)
        model.add(get_layer(layer))
    model.add(layers.Dense(units=num_classes, activation='softmax'))
    loss = getattr(tf.keras.losses, config['loss'])
    model.compile(
        optimizer=config.get('optimizer', 'adam'),
        loss=loss(),
        metrics=config['metrics'],
        # run_eagerly=True
    )
    return model


def fit_and_predict_dataset(config, num_classes, x_train, x_test, y_train, y_test=None, prefix=''):
    # with train dataset and test split
    model = get_model(config, num_classes)
    # early stopping
    early_stopping_loss = EarlyStopping(monitor = 'val_loss', patience = config['patience'])
    early_stopping_accuracy = EarlyStopping(monitor = 'val_accuracy', patience = config['patience'])
    # split train data into train and val
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=config['validation_split'], shuffle=True, stratify=y_train, random_state=42)
    # fit model
    history = model.fit(
        x_train, 
        y_train, 
        # validation_split=config['validation_split'], 
        validation_data = (x_val, y_val),
        epochs=config['epochs'],
        callbacks=[early_stopping_loss, early_stopping_accuracy]
    )

    with open(os.path.join(LOGS_PATH, f'{prefix}_model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # predict test from train dataset
    predictions = model.predict(x_test)
    preds = np.argmax(predictions, axis=1)

    test_df = pd.DataFrame()
    test_df['number'] = np.arange(len(preds))
    test_df['class'] = preds

    test_df.to_csv(os.path.join(LOGS_PATH, f'{prefix}_prediction_{exp}.csv'), index=False)

    pprint.pprint(history.history)
    
    if y_test is not None:
        print(metrics.classification_report(y_test, preds))
        with open(os.path.join(LOGS_PATH, 'classification_report.txt'), 'w') as file:
            file.write(metrics.classification_report(y_test, preds))

    with open(os.path.join(LOGS_PATH, f'{prefix}_history.json'), 'w') as outfile:
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

# main funtion of python
if __name__ == '__main__':
    set_global_determinism(42)
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

    width = 50
    height = 50
    num_classes = 6


    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=config['train_split'], shuffle=True, stratify=y, random_state=42)
        
    # # fit and predict train dataset
    print('PREDICT TRAIN DATASET')
    fit_and_predict_dataset(config.copy(), num_classes, x_train, x_test, y_train, y_test=y_test, prefix='test_split')
    # predict kaggle dataset
    print('PREDICT KAGGLE DATASET')
    print(config)
    fit_and_predict_dataset(config.copy(), num_classes, X, X_test, y, y_test=None, prefix='kaggle')
