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
# to ignore terminal warnings
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")


from sklearn.metrics import make_scorer


# plot parameters
plt.rcParams["figure.figsize"] = (20,12)
plt.rcParams['axes.grid'] = True
plt.style.use('fivethirtyeight')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['lines.linewidth'] = 3

def get_layer(layer_config):
    layer_name = layer_config.pop('type')
    layer_func = getattr(tf.keras.layers, layer_name)
    layer = layer_func(**layer_config)
    return layer

def get_model(config):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(50, 50, 3)))
    
    for layer in config["layers"]:
         model.add(get_layer(layer))

    loss = getattr(tf.keras.losses, config['loss'])
    model.compile(
        optimizer=config.get('optimizer', 'adam'),
        loss=loss(),
        metrics=config['metrics']
    )
    return model

# main funtion of python
if __name__ == '__main__':
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
    PREDICTIONS_PATH = os.path.join(LOGS_PATH, 'predictions')
    os.makedirs(PREDICTIONS_PATH)
    TARGET_PATH = os.path.join(LOGS_PATH, 'kaggle')
    os.makedirs(TARGET_PATH)

    # gets the name of the config file and read´s it
    config_path = sys.argv[1]
    with open(config_path, 'r') as file:
        config = json.load(file)
        
    # save config in logs folder
    with open(os.path.join(LOGS_PATH, 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=1)

    path = os.path.join('dataset')
    X = np.load(os.path.join(path, 'trainX.npy')) / 255
    y = np.load(os.path.join(path, 'trainy.npy'), allow_pickle=True).astype(int)
    X_test = np.load(os.path.join(path, 'testX.npy')) / 255

    width = 50
    height = 50
    num_classes = 6

    
    config = {
        'metrics' : ['accuracy'],
        'optimizer' : 'adam',
        'loss' : 'SparseCategoricalCrossentropy',
        'layers' : [
            {'type' : 'Flatten'},
            {'type':'Dense', 'units': 100, 'activation': 'relu'}
        ],
        'epochs' : 1,
        'validation_split' : 0.2
    }

    x_train, x_test, y_train, y_test = train_test_split(X, y)

    model = get_model(config)

    history = model.fit(
        x_train, 
        y_train, 
        validation_split=config['validation_split'], 
        epochs=config['epochs'])

    with open(os.path.join(LOGS_PATH, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    predictions = model.predict(x_test)
    preds = np.argmax(predictions, axis=1)

    print(metrics.classification_report(y_test, preds))
    pprint.pprint(history.history)
    
    with open(os.path.join(LOGS_PATH, 'classification_report.txt'), 'w') as file:
        file.write(metrics.classification_report(y_test, preds))


    with open(os.path.join(LOGS_PATH, 'history.json'), 'w') as outfile:
        json.dump(history.history, outfile, indent=1)
    
      
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
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
    plt.savefig(os.path.join(LOGS_PATH, 'train_results.png'))
    plt.show()