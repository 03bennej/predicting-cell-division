import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

tf.random.set_seed(42)

def LSTM(shape, dropout=0.8):
    inputs = keras.layers.Input(shape = shape)

    x = keras.layers.Bidirectional(keras.layers.LSTM(units=64, return_sequences=True, recurrent_dropout=dropout, dropout=dropout))(inputs)
    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(64, activation = 'relu')(x)
    x = keras.layers.Dropout(rate=dropout)(x)

    outputs = keras.layers.Dense(1, activation = 'sigmoid')(x)

    model = keras.Model(inputs, outputs, name="lstm")

    return model

def CNN(shape, dropout=0.8):
    inputs = keras.layers.Input(shape = shape)

    x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding="valid", activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(rate=dropout)(x)

    x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding="valid", activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(rate=dropout)(x)

    x = tf.keras.layers.Flatten()(x)

    x = keras.layers.Dense(units=64, activation="relu")(x)
    x = keras.layers.Dropout(rate=dropout)(x)

    outputs = keras.layers.Dense(1, activation = 'sigmoid')(x)

    model = keras.Model(inputs, outputs, name="cnn")

    return model

METRICS = [ 
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            ]       

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", 
                                                  mode="auto",
                                                  restore_best_weights=True, 
                                                  patience=1000)

compile_kwargs = {
                  "optimizer": keras.optimizers.Adam(learning_rate=0.0001),
                  "loss": keras.losses.BinaryCrossentropy(),
                  "metrics": METRICS
                  }

inner_fit_kwargs = {
              "epochs": 200,
              "batch_size": 128,
            #   "callbacks": [early_stopping]
              }