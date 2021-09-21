# This code was yoinked and reworked from here https://drive.google.com/file/d/1MOVhZhn0yv-Ngp0xK9jly-b_Ttx_2Tf7/view
# Seems to work on random inputs and outputs. Trimmed out a branch of this model that is not in my use case.
# Removed some weird loss function that I'm fairly certain was custom made for the authors expected outputs.

import keybindHandler
import tensorflow as tf
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



def train(path, chunk_size:int=None, model_save_name:str=None):

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

    #training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # For getting the sizes, we will read the first sample, then ignore it.
    dataLoader = DataLoader(path, normalize=True, chunk_size=chunk_size)
    data_sets = None
    for data_set in dataLoader:
        data_sets = data_set
        break

    """X2 = [(np.random.rand(1920 // 8, 1080 // 8, 3) * 255)//1 for i in range(100)]
    X2 = np.asarray(X2, dtype=np.float32)
    X2 = X2 / 255 # Normlization of image data

    X3 = [[[randint(0, 1) for w in range(20)] for j in range(5)] for i in range(100)]
    X3 = np.asarray(X3)

    y = [[randint(0, 1) for w in range(20)] for i in range(100)]"""

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
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    patience = 6
    timenow = str(datetime.datetime.now()).replace(":","_")
    earlystopper = EarlyStopping(monitor="f1", patience=patience, verbose=1, mode="max", restore_best_weights=True)
    if model_save_name is None:
        NAME = f"{layerSize}-nodes-{conv1Layer}-conv1-{conv2Layer}-conv2-{denseLayer+1}-dense-earlyStopping-patience-{patience}-time-{timenow}"
    else:
        NAME = model_save_name
    print(NAME)
    tensorBoard = TensorBoard(log_dir=f"logs/{NAME}")

    inputImage = Input(shape=x2_train[0].shape)
    inputHistory = Input(shape=x3_train[0].shape)

    # Image Branch
    y = Conv2D(layerSize, (3,3))(inputImage)
    y = Activation("relu")(y)
    y = MaxPooling2D(pool_size=(2,2))(y)

    y = Conv2D(layerSize, (3,3))(y)
    y = Activation("relu")(y)
    y = MaxPooling2D(pool_size=(2,2))(y)

    y = Flatten()(y)
    y = Model(inputs=inputImage, outputs=y)

    # History Branch
    w = LSTM(layerSize, input_shape=(1, x3_train.shape[1]), return_sequences=False)(inputHistory)
    w = Dropout(dropout)(w)
    w = Flatten()(w)
    w = Model(inputs=inputHistory, outputs=w)

    # Combine Inputs
    combined = concatenate([y.output, w.output])

    # Combined Branch
    z = Dense(256, activation="relu")(combined)
    z = Dropout(dropout)(z)

    z = Dense(layerSize, activation="relu")(z)
    z = Dropout(dropout)(z)

    # Output Layer
    z = Dense(len(keybindHandler.EMPTY_CLASSES_ONEHOT), activation="sigmoid")(z)
    model = Model(inputs=[y.input, w.input], outputs=z)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, decay=1e-5)
    def my_loss(targets, logits):
        weights = np.array([0.8 for _ in range(len(keybindHandler.EMPTY_CLASSES_ONEHOT))])
        return K.sum(targets * -K.log(1 - logits + 1e-10) * weights + (1 - targets) * -K.log(1 - logits + 1e-10) * (1 - weights), axis=-1)

    #opt = keras.optimizers.Adam(lr=1e-4, decay=1e-5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[f1])
    model.fit([x2_train, x3_train], y_train,\
            epochs=100, batch_size=20,\
            callbacks=[tensorBoard])
    #val_loss, val_acc = model.evaluate([x2_test, x3_test], y_test)
    #print(f"Validation_loss: {val_loss}, Validation accuracy: {val_acc}")

    print(f"Saving Model at path {NAME}.model...")
    model.save(f"{NAME}.model")
