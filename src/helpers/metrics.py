import time
import itertools
import pandas as pd
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')



class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


    def save(self, filename):
        print("\n Training time {} \n".format(self.times))
        time_callback= pd.DataFrame({'trainining time':  self.times})
        print("\nSaving training time...\n")
        time_callback.to_csv(filename, index=False)



class Metrics(Callback):

    def __init__(self, validation_generator, validation_steps=None, threshold=0.5):
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps or len(validation_generator)
        self.threshold = threshold


    def on_train_begin(self, logs={}):
        self.val_f1_scores = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):

        gen_1, gen_2 = itertools.tee(self.validation_generator)

        y_true = np.vstack(next(gen_1)[1] for _ in range(self.validation_steps)).astype('int')

        y_pred = (self.model.predict(gen_2, steps=self.validation_steps) > self.threshold).astype('int')
        _val_f1 = f1_score(y_true, y_pred, average='weighted')
        _val_recall = recall_score(y_true, y_pred, average='weighted')
        _val_precision = precision_score(y_true, y_pred, average='weighted')
        self.val_f1_scores.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(f" - val_f1_score: {_val_f1:.5f} - val_precision: {_val_precision:.5f} - val_recall: {_val_recall:.5f}")
        return


    def save(self, filename):

        metrics = pd.DataFrame({
            'f1-score': self.val_f1_scores,
            'recall': self.val_recalls,
            'precision': self.val_precisions
        })
        print("\nSaving metrics...\n")
        metrics.to_csv(filename, index=False)
