# AML project: Image classification

## Authors

- Joana Simões
- Pedro Carrasco

## Run the program

    cd src
    python main.py config.json

The structure of the config.json file is described below

## Configuration file

The configuration file should have the following parameters in a json format:

| Parameter | Description | Default Value | Type |
| :--- | :---: | :---: | ---: |
| metrics | Metric to be optimized by the keras during training | ["accuracy"] | list of strings |
| optimizer | Name of the optimizer to use. Is should have the same name as the Keras optimizers instances | "Adam" | string |
| learning_rate | Learning rate to use by the optimizer | 0.001 | float |
| loss | Loss Functions. Is should have the same name as the Keras instances | "SparseCategoricalCrossentropy" | string |
| layers | List of layers for the models. The type key should have the same value as the Keras layer's name | - | list of dictionaries |
| epochs | Number of epochs to train the models | 200 | int |
| validation_split | Percentage of the train dataset for validation | 0.2 | float |
| training_split | Percentage of the dataset for training | 0.8 | float |
| patience | Number of epochs to wait before stopping training after no improvements| 10 | int |
| batch_size | Number of sample in each batch | 32 | int |
| cross-validation | Either or not to use cross-validation | false | boolean |
| predict-kaggle | Either or not to re-train the model to predict the kaggle dataset | false | boolean |
| ensemble | Either or not to use ensemble of the prediction (train the model with different parts of the dataset) | false | boolean |
| ensemble_n_estimators | Number of split to use for ensemble (in Stratified K-fold)| 5 | int |
|predict_binary | Either or not to use binary classifier to help in the samples with most difficulty | false | boolean |
| only-kaggle | Either or not to only train and predict for the kaggle competition | false | boolean |
| deep-model | To use "xception", "resnet50", "vgg16" or null. It overrides the layers parameters | vgg16 | string |
