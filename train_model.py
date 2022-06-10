import numpy as np
import functions
import tensorflow as tf

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


def get_time():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return run_id


def train_on_all_data(modelpath):
    X, y = functions.get_sequences_and_targets(functions.get_all_data_for_train())
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    class_w = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {0: class_w[0], 1: class_w[1]}

    early_stopping = EarlyStopping(restore_best_weights=True, patience=1000)
    model = tf.keras.models.load_model(modelpath)

    model.fit(X_train,
              y_train,
              epochs=3000,
              batch_size=20,
              validation_data=(X_test, y_test),
              class_weight=class_weights,
              callbacks=[early_stopping])

    model.evaluate(X_test, y_test)

    model.save(f'.\models\{model.name}_{get_time()}')


def train(filepath, modelpath):
    X, y = functions.get_sequences_and_targets(functions.get_data_for_train(filepath))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    class_w = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {0: class_w[0], 1: class_w[1]}

    early_stopping = EarlyStopping(restore_best_weights=True, patience=1000)
    model = tf.keras.models.load_model(modelpath)

    model.fit(X_train,
              y_train,
              epochs=3000,
              batch_size=20,
              validation_data=(X_test, y_test),
              class_weight=class_weights,
              callbacks=[early_stopping])

    model.evaluate(X_test, y_test)

    model.save(f'.\models\{model.name}_{get_time()}')
