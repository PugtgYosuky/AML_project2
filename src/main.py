"""
        ACA Project

        AUTHORS:
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
    """
    Fix seeds for reproducibility
    """
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
    """
    Converts a layers form the config file into a keras layer
    """
    layer_config = layer_config.copy()
    layer_name = layer_config.pop('type')
    layer_func = getattr(tf.keras.layers, layer_name)
    layer = layer_func(**layer_config)
    return layer

def get_model(config, num_classes, width=50, height=50):
    """
    Instantiates a keras model
    """
    loss = getattr(tf.keras.losses, config['loss'])
    optimizer = getattr(tf.keras.optimizers.legacy, config['optimizer'])
    model_name = config.get('deep-model', None)
    
    model = models.Sequential()
    base_model = None
    if model_name is None:
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(width, height, 3)))
        for layer in config["layers"]:
            model.add(get_layer(layer))
        model.add(layers.Dense(units=num_classes, activation='softmax'))
    else:
        model.add(layers.InputLayer(input_shape=(width, height, 3)))
        model.add(layers.RandomFlip(mode='horizontal'))
        model.add(layers.RandomBrightness(factor=0.2, value_range=(0.0, 1.0)))
        model.add(layers.RandomContrast(factor=0.2))
        model.add(layers.RandomRotation(factor=0.2))
        if model_name == 'vgg16':
            base_model = VGG16(
                weights='imagenet', 
                include_top=False,
                pooling = 'max',
                input_shape=(width, height, 3),
                )
        elif model_name == 'resnet50': 
            base_model = ResNet50(
                weights='imagenet', 
                include_top=False,
                pooling = 'max',
                input_shape=(width, height, 3),
                )
        else:
            model.add(layers.Resizing(width=72, height=72))
            base_model = Xception(
                weights='imagenet', 
                include_top=False,
                pooling = 'max',
                input_shape=(72, 72, 3),
                )  
        base_model.trainable = False
        model.add(base_model)
    model.add(layers.Dense(units=num_classes, activation='softmax'))
    model.compile(
        optimizer=optimizer(learning_rate=config['learning_rate']),
        loss=loss(),
        metrics=config['metrics'],
    )
    print(model.summary())
    return model, base_model

def write_predictions(preds):
    # writes predictions
    test_df = pd.DataFrame()
    test_df['number'] = np.arange(len(preds))
    test_df['class'] = preds
    return test_df

def fit_and_predict_dataset(config, num_classes, x_train, x_test, y_train, y_test=None, prefix='', path='', seed=42, x_val=None, y_val=None):
    """
    Train and predict given a certain dataset
    """

    # with train dataset and test split
    model, base_model = get_model(config, num_classes)

    # early stopping
    early_stopping_loss = EarlyStopping(
        monitor = 'val_loss', 
        patience = config['patience'], 
        restore_best_weights = True,
        )
    early_stopping_accuracy = EarlyStopping(monitor = 'val_accuracy', patience = config['patience'], mode='max', restore_best_weights=True)
    # split train data into train and val
    if x_val is None:
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

    # fine-tune the model
    if config.get('deep-model', None) is not None:
        with open(os.path.join(path, f'{prefix}_history-prev-finetuning.json'), 'w') as outfile:
            json.dump(history.history, outfile, indent=1)
        print('Fine-tunning')
        base_model.trainable = True

        model.compile(
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5),
            loss = tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics = config['metrics'],
        )
        early_stopping_loss = EarlyStopping(
            monitor = 'val_loss', 
            patience = 10, 
            restore_best_weights = True,
            )
        history = model.fit(
            x_train, 
            y_train, 
            # validation_split=config['validation_split'], 
            validation_data = (x_val, y_val),
            epochs=2,
            callbacks=[early_stopping_loss],
            batch_size=config['batch_size']
        )

    with open(os.path.join(path, f'{prefix}_history.json'), 'w') as outfile:
        json.dump(history.history, outfile, indent=1)

    model.save(os.path.join(path, f'{prefix}_model.h5'))
    with open(os.path.join(LOGS_PATH, f'{prefix}_model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # predict test from train dataset
    predictions = model.predict(x_test)
    preds = np.argmax(predictions, axis=1)
    
    # predict the worst classes with the binary classifier
    if config.get('predict_binary', False):
        classes = [[0, 5], [2, 3], [2, 4]]
        for class_a, class_b in classes:
            print(class_a)
            print(class_b)
            indexes = (preds == class_a) | (preds == class_b)
            test_classes = x_test[indexes]
            preds_bin = binary_classifier(class_a, class_b, test_classes)
            preds[indexes] = preds_bin.T[:, 0]

    test_df = write_predictions(preds)

    pprint.pprint(history.history)
    
    if y_test is not None:
        print(metrics.classification_report(y_test, preds))
        with open(os.path.join(path, f'{prefix}_classification_report.txt'), 'w') as file:
            file.write(metrics.classification_report(y_test, preds))
        test_df['real'] = y_test
        # save predictions
        test_df.to_csv(os.path.join(path, f'{prefix}_prediction_{exp}.csv'), index=False)
        # confusion matrix 
        cm = metrics.confusion_matrix(y_test, preds)
        # save confusion matrix as image
        plt.figure()
        disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'])
        disp.plot()
        plt.title('Confusion matrix')
        plt.savefig(os.path.join(LOGS_PATH, f'{prefix}_confusion_matrix.png'))
    else:
        test_df.to_csv(os.path.join(path, f'{prefix}_prediction_{exp}.csv'), index=False)


    # get metrics
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
    # ensemble by training the same model with different parts of the dataset
    predictions = pd.DataFrame()
    kfold = StratifiedKFold(n_splits=config['ensemble_n_estimators'], shuffle=True, random_state=seed)
    for i, (train_indexes, val_indexes) in enumerate(kfold.split(x_train, y_train)):
        # with replacement
        print('Ensemble model', i)
        x_train_aux = x_train[train_indexes]
        y_train_aux = y_train[train_indexes]
        x_val_aux = x_train[val_indexes]
        y_val_aux = y_train[val_indexes]
        preds = fit_and_predict_dataset(config.copy(), num_classes, x_train_aux, x_test, y_train_aux, y_test=y_test, prefix=f'{prefix}_{i}',path=path, seed=seed, x_val=x_val_aux, y_val=y_val_aux)
        predictions[f'model_{i}'] = preds

    p = predictions.mode(axis=1).to_numpy()[:, 0].astype(int)
    return p

def get_binary_model():
    # binary models
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(50, 50, 3)))
    model.add(layers.Conv2D(16, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics='accuracy',
        run_eagerly=True
    )
    return model

def binary_classifier(class_a, class_b, x_test):
    model = models.load_model(f'binary_models/{class_a}_{class_b}_model.h5')
    predictions = model.predict(x_test)
    predictions[predictions > 0.5] = class_b
    predictions[predictions <= 0.5] = class_a
    print(class_a, class_b)
    print(np.unique(predictions))

    return predictions

# main function of python
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
    if config.get('only-kaggle', False) == False:
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
