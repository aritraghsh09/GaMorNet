from keras.models import Sequential, clone_model
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import VarianceScaling
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras import backend as K
from keras.engine.topology import Layer
import wget
import random
import string
import os
import numpy as np

############################################
##########HELPER FUNCTIONS##################


def check_input_shape_validity(input_shape):

    if input_shape == 'SDSS':
        input_shape = (167, 167, 1)
    elif input_shape == 'CANDELS':
        input_shape = (83, 83, 1)
    else:
        try:
            if len(input_shape) != 3:
                raise Exception(
                    "input_shape has to be a tuple (typically of 3 dimensions) or any of the available keywords")
        except:
            raise Exception(
                "input_shape has to be a tuple (typically of 3 dimensions) or any of the available keywords")

    return input_shape


def check_imgs_validity(img_array):

    if isinstance(img_array, np.ndarray):
        if len(img_array.shape) != 4:
            raise Exception(
                "The Image Array needs to have 4 dimensions. (num,x,y,bands)")
    else:
        raise Exception(
            "The Image Array Needs to be a 4 Dimensional Numpy Array. (num,x,y,bands)")


def check_labels_validity(labels):

    if isinstance(labels, np.ndarray):
        if labels.shape[1] != 3:
            raise Exception(
                "The Labels Array needs to have 2 dimensions. (num,3)")
    else:
        raise Exception(
            "The Lables Array Needs to be a 2 Dimensional Numpy Array. (num,3)")


def check_bools_validity(bools):

    if (bools == 'train_bools_SDSS'):
        bools = [True]*8
    elif (bools == 'train_bools_CANDELS'):
        bools = [False, False, False, True, True, True, True, True]
    elif (bools == 'load_bools_SDSS'):
        bools = [True, True, True, True, True, False, False, False]
    elif (bools == 'load_bools_CANDELS'):
        bools = [True, True, True, True, True, True, False, False]

    try:
        for element in bools:
            if type(element) != bool:
                raise Exception(
                    "The Supplied Array of Bools doesn't look okay")

        if len(bools) != 8:
            raise Exception("The Supplied Array of Bools doesn't look okay")

    except:
        raise Exception("The Supplied Array of Bools doesn't look okay")

    return bools


def get_model_from_link_keras(link, model):

    letters = string.ascii_lowercase
    if link[-4:] == 'hdf5':
        file_name = ''.join(random.choice(letters)
                            for i in range(15)) + '.hdf5'
    else:
        file_name = ''.join(random.choice(letters) for i in range(15)) + '.h5'

    wget.download(link, out=file_name)

    try:
        model.load_weights(file_name)
    except:
        os.remove(file_name)
        raise

    os.remove(file_name)

    return model


def gamornet_load_model_keras(model, model_load_path):

    if model_load_path == 'SDSS_sim':
        print("Fetching SDSS Sim Trained Weigths.....")
        link = 'ftp://ftp.astro.yale.edu/pub/aghosh/gamornet_keras/trained_models/SDSS/sim_trained/model_sdss_sim_trained.h5'
        model = get_model_from_link_keras(link, model)

    elif model_load_path == 'SDSS_tl':
        print("Fetching SDSS TL Weigths.....")
        link = 'ftp://ftp.astro.yale.edu/pub/aghosh/gamornet_keras/trained_models/SDSS/tl/model_sdsss_tl.hdf5'
        model = get_model_from_link_keras(link, model)

    elif model_load_path == 'CANDELS_sim':
        print("Fetching CANDELS Sim Trained Weigths.....")
        link = 'ftp://ftp.astro.yale.edu/pub/aghosh/gamornet_keras/trained_models/CANDELS/sim_trained/model_candels_sim_trained.hdf5'
        model = get_model_from_link_keras(link, model)

    elif model_load_path == 'CANDELS_tl':
        print("Fetching CANDELS TL Weigths.....")
        link = 'ftp://ftp.astro.yale.edu/pub/aghosh/gamornet_keras/trained_models/CANDELS/tl/model_candels_tl.hdf5'
        model = get_model_from_link_keras(link, model)

    else:
        model.load_weights(model_load_path)

    return model


###########################################
###########################################


############################################
##########KERAS FUNCTIONS##################

# Implementing LRN in Keras. Code from "Deep Learning with Python" by F. Chollet
class LocalResponseNormalization(Layer):

    def __init__(self, n=5, alpha=0.0001, beta=0.75, k=1.0, **kwargs):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LocalResponseNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        super(LocalResponseNormalization, self).build(input_shape)

    def call(self, x, mask=None):
        if K.image_data_format == "channels_first":
            _, f, r, c = self.shape
        else:
            _, r, c, f = self.shape
        squared = K.square(x)
        pooled = K.pool2d(squared, (self.n, self.n), strides=(1, 1),
                          padding="same", pool_mode="avg")
        if K.image_data_format == "channels_first":
            summed = K.sum(pooled, axis=1, keepdims=True)
            averaged = self.alpha * K.repeat_elements(summed, f, axis=1)
        else:
            summed = K.sum(pooled, axis=3, keepdims=True)
            averaged = self.alpha * K.repeat_elements(summed, f, axis=3)
        denom = K.pow(self.k + averaged, self.beta)
        return x / denom

    def get_output_shape_for(self, input_shape):
        return input_shape


def gamornet_build_model_keras(input_shape):

    input_shape = check_input_shape_validity(input_shape)

    # uniform scaling initializer
    uniform_scaling = VarianceScaling(
        scale=1.0, mode='fan_in', distribution='uniform', seed=None)

    # Building GaMorNet
    model = Sequential()

    model.add(Conv2D(96, 11, strides=4, activation='relu', input_shape=input_shape,
                     padding='same', kernel_initializer=uniform_scaling))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
    model.add(LocalResponseNormalization())

    model.add(Conv2D(256, 5, activation='relu', padding='same',
                     kernel_initializer=uniform_scaling))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
    model.add(LocalResponseNormalization())

    model.add(Conv2D(384, 3, activation='relu', padding='same',
                     kernel_initializer=uniform_scaling))
    model.add(Conv2D(384, 3, activation='relu', padding='same',
                     kernel_initializer=uniform_scaling))
    model.add(Conv2D(256, 3, activation='relu', padding='same',
                     kernel_initializer=uniform_scaling))
    model.add(MaxPooling2D(pool_size=3, strides=2, padding='same'))
    model.add(LocalResponseNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='tanh',
                    kernel_initializer='TruncatedNormal'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='tanh',
                    kernel_initializer='TruncatedNormal'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax',
                    kernel_initializer='TruncatedNormal'))

    return model


def gamornet_predict_keras(img_array, model_load_path, input_shape, batch_size=64, individual_arrays=False):

    check_imgs_validity(img_array)

    model = gamornet_build_model_keras(input_shape=input_shape)

    print("Loading GaMorNet Model.....")
    model = gamornet_load_model_keras(model, model_load_path)

    print("Performing Predictions.....")
    preds = model.predict(img_array, batch_size=batch_size)
    preds = np.array(preds)  # converting to a numpy array for easier handling.

    if individual_arrays is True:
        # 'disk_prob','unclass_prob','bulge_prob'
        return preds[:, 0], preds[:, 1], preds[:, 2]
    else:
        return preds


def gamornet_train_keras(training_imgs, training_labels, validation_imgs, validation_labels, input_shape, files_save_path="./", epochs=100, checkpoint_freq=0, batch_size=64, lr=0.0001, momentum=0.9, decay=0.0, nesterov=False, loss='categorical_crossentropy', load_model=False, model_load_path="./", save_model=True, verbose=1):

    check_imgs_validity(training_imgs)
    check_imgs_validity(validation_imgs)
    check_labels_validity(training_labels)
    check_labels_validity(validation_labels)

    model = gamornet_build_model_keras(input_shape=input_shape)

    sgd = optimizers.SGD(lr=lr, momentum=momentum,
                         decay=decay, nesterov=nesterov)
    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])

    callbacks_list = []

    if checkpoint_freq != 0:
        checkpoint = ModelCheckpoint(files_save_path + 'model_{epoch:02d}.hdf5', monitor='val_loss',
                                     verbose=verbose, save_best_only=False, save_weights_only=False, mode='auto', period=checkpoint_freq)
        callbacks_list.append(checkpoint)

    csv_logger = CSVLogger(files_save_path + "metrics.csv",
                           separator=',', append=False)
    callbacks_list.append(csv_logger)

    if load_model is True:
        model = gamornet_load_model_keras(model, model_load_path)

    model.fit(training_imgs, training_labels, batch_size=batch_size, epochs=epochs, verbose=verbose,
              validation_data=(validation_imgs, validation_labels), shuffle=True, callbacks=callbacks_list)

    if save_model is True:
        model.save(files_save_path + "trained_model.hdf5")

    return model


def gamornet_tl_keras(training_imgs, training_labels, validation_imgs, validation_labels, input_shape, load_layers_bools=[True]*8, trainable_bools=[True]*8, model_load_path="./", files_save_path="./", epochs=100, checkpoint_freq=0, batch_size=64, lr=0.00001, momentum=0.9, decay=0.0, nesterov=False, loss='categorical_crossentropy', save_model=True, verbose=1):

    check_imgs_validity(training_imgs)
    check_imgs_validity(validation_imgs)
    check_labels_validity(training_labels)
    check_labels_validity(validation_labels)
    load_layers_bools = check_bools_validity(load_layers_bools)
    trainable_bools = check_bools_validity(trainable_bools)

    model = gamornet_build_model_keras(input_shape=input_shape)
    model_new = clone_model(model)
    model = gamornet_load_model_keras(model, model_load_path)

    # Reversing the Order of the Bools because I will call .pop() on these later
    load_layers_bools.reverse()
    trainable_bools.reverse()

    for i in range(len(model_new.layers)):

        if model_new.layers[i].count_params() != 0:

            model_new.layers[i].trainable = trainable_bools.pop()

            if load_layers_bools.pop() is True:
                model_new.layers[i].set_weights(model.layers[i].get_weights())
                print("Loading Layer" + str(i) + " from previous model.")
            else:
                print("Initializing Layer" + str(i) + " from scratch")

        else:
            model_new.layers[i].set_weights(model.layers[i].get_weights())

    sgd = optimizers.SGD(lr=lr, momentum=momentum,
                         decay=decay, nesterov=nesterov)
    model_new.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])

    callbacks_list = []

    if checkpoint_freq != 0:
        checkpoint = ModelCheckpoint(files_save_path + 'model_{epoch:02d}.hdf5', monitor='val_loss',
                                     verbose=verbose, save_best_only=False, save_weights_only=False, mode='auto', period=checkpoint_freq)
        callbacks_list.append(checkpoint)

    csv_logger = CSVLogger(files_save_path + "metrics.csv",
                           separator=',', append=False)
    callbacks_list.append(csv_logger)

    model_new.fit(training_imgs, training_labels, batch_size=batch_size, epochs=epochs, verbose=verbose,
                  validation_data=(validation_imgs, validation_labels), shuffle=True, callbacks=callbacks_list)

    if save_model is True:
        model_new.save(files_save_path + "trained_model.hdf5")

    return model_new


###########################################
###########################################
