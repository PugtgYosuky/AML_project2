import pandas as pd
import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt



if __name__ == '__main__':
    experience = 'exp76'
    predictions = pd.read_csv(os.path.join('logs', experience, f'test_split_prediction_{experience}.csv'))
    cm = metrics.confusion_matrix(predictions.real, predictions['class'])

    # save confusion matrix as image
    plt.figure()
    disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'])
    disp.plot()
    plt.title('Confusion matrix')
    plt.show()