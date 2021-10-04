# This code was yoinked and reworked from here https://drive.google.com/file/d/1MOVhZhn0yv-Ngp0xK9jly-b_Ttx_2Tf7/view
# Seems to work on random inputs and outputs. Trimmed out a branch of this model that is not in my use case.
# Removed some weird loss function that I'm fairly certain was custom made for the authors expected outputs.

from stateManager import HISTORY_LENGTH, MAX_CHUNK_SIZE
import keybindHandler
import tensorflow as tf
import numpy as np
import keras
import datetime
import os
from random import randint, shuffle
from keras.layers import Dense, Dropout, LSTM, Flatten, Input, concatenate, TimeDistributed, MaxPooling2D, MaxPool2D, GlobalMaxPool2D, Conv2D, BatchNormalization, ConvLSTM2D
from keras.models import Model
import keras.metrics
from tensorflow.keras.utils import Sequence
from keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.applications.efficientnet import EfficientNetB0
import keras.backend as K
from dataLoader2 import DataLoader


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

def create_data_generator(path, chunk_size):
    def dl_generator():
        dataLoader = DataLoader(path, normalize=True, chunk_size=chunk_size)
        while True:
            for data_set in dataLoader:
                X2 = data_set[0]
                X2 = np.asarray(X2, dtype=np.float32)
                X2 = X2 / 255 # Normlization of image data

                X3 = data_set[1]
                X3 = np.asarray(X3)

                y = data_set[2]
                y = np.asarray(y, dtype=np.float32)

                yield (X2, y)
            dataLoader = DataLoader(path, normalize=True, chunk_size=chunk_size)
    return dl_generator

def build_convnet_eff(shape=(112, 112, 3)):
    model = keras.Sequential()
    tf.keras.applications.efficientnet.EfficientNetB0()
    model.add(Conv2D(64, (3,3), input_shape=shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3,3), input_shape=shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
 
    # flatten...
    model.add(GlobalMaxPool2D())
    return model

def build_convnet(shape=(112, 112, 3)):
    momentum = .9
    model = keras.Sequential()
    model.add(Conv2D(32, (3,3), input_shape=shape,
        padding='same', activation='relu'))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    # flatten...
    model.add(GlobalMaxPool2D())
    return model

def build_model_non_stateful(input_shape=(1,1920,1080,3)):
    base_model = EfficientNetB0(weights='imagenet',input_shape=(input_shape[1:]),include_top=False,drop_connect_rate=0.2)
    base_model.trainable = True
    intermediate_model= Model(inputs=base_model.input, outputs=base_model.layers[161].output)
    intermediate_model.trainable = True
    input_1 = Input(shape=input_shape,name='main_in')
    x = TimeDistributed(intermediate_model)(input_1)
    x = ConvLSTM2D(filters=128,kernel_size=(3,3),stateful=False,return_sequences=True)(x)
    x = TimeDistributed(Flatten())(x)
    output_1 = TimeDistributed(Dense(keybindHandler.ACTION_CLASS_SIZE, activation='sigmoid'))(x)
    output_2 = TimeDistributed(Dense(keybindHandler.MOUSE_CLASS_SIZE, activation='softmax'))(x)
    output_3 = TimeDistributed(Dense(keybindHandler.MOUSE_CLASS_SIZE, activation='softmax'))(x) 
    output_4 = TimeDistributed(Dense(1, activation='linear'))(x) 
    output_all = concatenate([output_1,output_2,output_3,output_4], axis=-1)
    model = Model(input_1, output_all)
    return model


def train(path, chunk_size:int=None, model_save_name:str=None):

    # For getting the sizes, we will read the first sample, then ignore it.
    dataLoader = DataLoader(path, normalize=True, chunk_size=HISTORY_LENGTH)
    number_of_samples = dataLoader.get_number_of_samples()
    print(f"DataLoader has {number_of_samples} samples.")
    data_sets = None
    for data_set in dataLoader:
        data_sets = data_set
        break
    dataLoader.the_file.close()
    dataLoader = None

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

    x2_train = X2
    x3_train = X3
    y_train = y



    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    patience = 6
    timenow = str(datetime.datetime.now()).replace(":","_")
    earlystopper = EarlyStopping(monitor="f1", patience=patience, verbose=1, mode="max", restore_best_weights=True)
    if model_save_name is None:
        NAME = f"csgopapermodel-earlyStopping-patience-{patience}-time-{timenow}"
    else:
        NAME = model_save_name
    print(NAME)
    tensorBoard = TensorBoard(log_dir=f"logs/{NAME}")

    model = build_model_non_stateful(x2_train[0].shape)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, decay=1e-5)
    #opt = keras.optimizers.Adam(lr=1e-4, decay=1e-5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[f1])

    dl_gen = create_data_generator(path, chunk_size=5)

    model.fit(dl_gen(),\
            epochs=64, steps_per_epoch=number_of_samples // 5,\
            callbacks=[tensorBoard, earlystopper])

    """for epoch in range(100):
        # Create a new data loader for the max chunk size
        dataLoader = DataLoader(path, normalize=True, chunk_size=MAX_CHUNK_SIZE)
        for data_set in dataLoader:
            X2, X3, y = generate_data(data_set)
            model.fit([X2, X3], y, batch_size=20, epochs=epoch+1, initial_epoch=epoch, shuffle=True, callbacks=[tensorBoard])"""

    """model.fit([x2_train, x3_train], y_train,\
            epochs=100, batch_size=20, shuffle=False,\
            callbacks=[tensorBoard])"""
    #val_loss, val_acc = model.evaluate([x2_test, x3_test], y_test)
    #print(f"Validation_loss: {val_loss}, Validation accuracy: {val_acc}")

    print(f"Saving Model at path {NAME}.model...")
    model.save(f"{NAME}.model")
