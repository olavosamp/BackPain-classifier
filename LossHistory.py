# Callback to record train and validation history
import keras.callbacks

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.valLosses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.valLosses.append(logs.get('val_loss'))