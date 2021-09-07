# This code was yoinked and reworked from here https://drive.google.com/file/d/1MOVhZhn0yv-Ngp0xK9jly-b_Ttx_2Tf7/view
# Seems to work on random inputs and outputs. Trimmed out a branch of this model that is not in my use case.
# Removed some weird loss function that I'm fairly certain was custom made for the authors expected outputs.

from dataEncoder import BLANK_CLASS_OUTPUT
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
import datetime
import os
from random import randint
from keras.layers import Dense, Dropout, CuDNNLSTM, LSTM, Flatten, Input,\
Activation, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, concatenate
from keras.models import Model
import keras.metrics
from keras.callbacks import TensorBoard, EarlyStopping
import keras.backend as K
from dataLoader import DataLoader



def re_train(path, model_load_name:str, chunk_size:int=None, model_save_name:str=None):

    # This metric was taken from here: https://drive.google.com/file/d/1MOVhZhn0yv-Ngp0xK9jly-b_Ttx_2Tf7/view
    # by the author of said paper who took it from a stackoverflow post listed there
    def f1(y_true, y_pred):
        def recall(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall
        def precision(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (possible_positives + K.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


    dataLoader = DataLoader(path, normalize=True, chunk_size=chunk_size)
    data_sets = None
    for data_set in dataLoader:
        data_sets = data_set
        break

    X2 = data_sets[0]
    X2 = np.asarray(X2, dtype=np.float32)
    X2 = X2 / 255 # Normlization of image data

    X3 = data_sets[1]
    X3 = np.asarray(X3)

    y = data_sets[2]

    print(f"X2.shape = {X2.shape}")
    print(f"X3.shape = {X3.shape}")

    y = np.asarray(y, dtype=np.float32)

    trainingFraction = 0.2

    trainingSize = int(len(X2) * trainingFraction)

    x2_train = X2[:trainingSize]
    x3_train = X3[:trainingSize]
    y_train = y[:trainingSize]

    x2_test = X2[trainingSize:]
    x3_test = X3[trainingSize:]
    y_test = y[trainingSize:]



    layerSize = 128
    conv1Layer = 2
    conv2Layer = 3
    denseLayer = 2
    dropout = 0.3

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    patience = 6
    timenow = str(datetime.datetime.now()).replace(":","_")
    earlystopper = EarlyStopping(monitor="f1", patience=patience, verbose=1, mode="max", restore_best_weights=True)
    if model_save_name is None:
        NAME = f"{layerSize}-nodes-{conv1Layer}-conv1-{conv2Layer}-conv2-{denseLayer+1}-dense-earlyStopping-patience-{patience}-time-{timenow}"
    else:
        NAME = model_save_name
    print(NAME)

    tensorBoard = TensorBoard(log_dir=f"logs/{NAME}")

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, decay=1e-5)
    def my_loss(targets, logits):
        weights = np.array([0.8 for _ in range(len(BLANK_CLASS_OUTPUT))])
        return K.sum(targets * -K.log(1 - logits + 1e-10) * weights + (1 - targets) * -K.log(1 - logits + 1e-10) * (1 - weights), axis=-1)

    model = keras.models.load_model(model_load_name, custom_objects={"f1": f1, "my_loss": my_loss})

    #opt = keras.optimizers.Adam(lr=1e-4, decay=1e-5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[f1])
    model.fit([x2_train, x3_train], y_train,\
            epochs=40, batch_size=64,\
            callbacks=[tensorBoard, earlystopper])

    val_loss, val_acc = model.evaluate([x2_test, x3_test], y_test)
    print(f"Validation_loss: {val_loss}, Validation accuracy: {val_acc}")

    print(f"Saving Model at path {NAME}.model...")
    model.save(f"{NAME}.model")
