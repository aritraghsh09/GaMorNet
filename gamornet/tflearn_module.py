import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.optimizers import Nesterov, Momentum
from keras import backend as K
import wget
import os
import numpy as np
import progressbar
import time

############################################
##########HELPER FUNCTIONS##################

from gamornet.keras_module import check_input_shape_validity, check_imgs_validity, check_labels_validity, check_bools_validity


def get_model_from_link_tflearn(base_link, file_names, model):

    # Skipping over random name generation because the tflearn files themselves are named in a super specific way
    # Thus there is no chance of accidental deletion like the keras model.h5 names

    file_links = np.core.defchararray.add(np.array([base_link]*3), file_names)

    for link in file_links:
        wget.download(link)

    try:
        model.load(file_names[2][:-5])
    except:
        os.remove(file_names[0])
        os.remove(file_names[1])
        os.remove(file_names[2])
        raise

    os.remove(file_names[0])
    os.remove(file_names[1])
    os.remove(file_names[2])

    return model


def gamornet_load_model_tflearn(model, model_load_path):

    if model_load_path == 'SDSS_sim':
        print("Fetching SDSS Sim Trained Weigths.....")
        base_link = 'ftp://ftp.astro.yale.edu/pub/aghosh/gamornet/trained_models/SDSS/sim_trained/'
        file_names = np.array(
            ['check-1405593.data-00000-of-00001', 'check-1405593.index', 'check-1405593.meta'])
        model = get_model_from_link_tflearn(base_link, file_names, model)

    elif model_load_path == 'SDSS_tl':
        print("Fetching SDSS TL Weigths.....")
        base_link = 'ftp://ftp.astro.yale.edu/pub/aghosh/gamornet/trained_models/SDSS/tl/'
        file_names = np.array(
            ['check-1546293.data-00000-of-00001', 'check-1546293.index', 'check-1546293.meta'])
        model = get_model_from_link_tflearn(base_link, file_names, model)

    elif model_load_path == 'CANDELS_sim':
        print("Fetching CANDELS Sim Trained Weigths.....")
        base_link = 'ftp://ftp.astro.yale.edu/pub/aghosh/gamornet/trained_models/CANDELS/sim_trained/'
        file_names = np.array(
            ['check-562800.data-00000-of-00001', 'check-562800.index', 'check-562800.meta'])
        model = get_model_from_link_tflearn(base_link, file_names, model)

    elif model_load_path == 'CANDELS_tl':
        print("Fetching CANDELS TL Weigths.....")
        base_link = 'ftp://ftp.astro.yale.edu/pub/aghosh/gamornet/trained_models/CANDELS/tl/'
        file_names = np.array(
            ['check-571275.data-00000-of-00001', 'check-571275.index', 'check-571275.meta'])
        model = get_model_from_link_tflearn(base_link, file_names, model)

    else:
        model.load(model_load_path)

    return model


###########################################
###########################################


############################################
##########TFLEARN FUNCTIONS##################

def gamornet_build_model_tflearn(input_shape, trainable_bools=[True]*8, load_layers_bools=[True]*8):

    input_shape = check_input_shape_validity(input_shape)
    load_layers_bools = check_bools_validity(load_layers_bools)
    trainable_bools = check_bools_validity(trainable_bools)

    model = input_data(shape=[None, input_shape[0],
                              input_shape[1], input_shape[2]])
    model = conv_2d(model, 96, 11, strides=4, activation='relu',
                    trainable=trainable_bools[0], restore=load_layers_bools[0])
    model = max_pool_2d(model, 3, strides=2)
    model = local_response_normalization(model)
    model = conv_2d(model, 256, 5, activation='relu',
                    trainable=trainable_bools[1], restore=load_layers_bools[1])
    model = max_pool_2d(model, 3, strides=2)
    model = local_response_normalization(model)
    model = conv_2d(model, 384, 3, activation='relu',
                    trainable=trainable_bools[2], restore=load_layers_bools[2])
    model = conv_2d(model, 384, 3, activation='relu',
                    trainable=trainable_bools[3], restore=load_layers_bools[3])
    model = conv_2d(model, 256, 3, activation='relu',
                    trainable=trainable_bools[4], restore=load_layers_bools[4])
    model = max_pool_2d(model, 3, strides=2)
    model = local_response_normalization(model)
    model = fully_connected(model, 4096, activation='tanh',
                            trainable=trainable_bools[5], restore=load_layers_bools[5])
    model = dropout(model, 0.5)
    model = fully_connected(model, 4096, activation='tanh',
                            trainable=trainable_bools[6], restore=load_layers_bools[6])
    model = dropout(model, 0.5)
    model = fully_connected(model, 3, activation='softmax',
                            trainable=trainable_bools[7], restore=load_layers_bools[7])

    return model


def gamornet_predict_tflearn(img_array, model_load_path, input_shape, batch_size=64, individual_arrays=False, trainable_bools=[True]*8, clear_session=False):

    # TFLearn Loads graphs from memory by name, hence it's always advisable to set this to True if using in a Notebook.
    if clear_session is True:
        K.clear_session()

    check_imgs_validity(img_array)

    model = gamornet_build_model_tflearn(
        input_shape=input_shape, trainable_bools=trainable_bools)
    model = tflearn.DNN(model)

    print("Loading GaMorNet Model.....")
    model = gamornet_load_model_tflearn(model, model_load_path)

    print("Performing Predictions.....")

    preds = []  # array to store results

    total_elements = len(img_array)
    num_batches = int(total_elements/batch_size)

    # Time to Flush all Print Statements before the progressbar output comes on
    time.sleep(0.1)
    for i in progressbar.progressbar(range(0, num_batches)):
        ll = i*batch_size
        ul = (i+1)*batch_size
        preds.extend(model.predict(img_array[ll:ul]))

    if ul != len(img_array):
        # for the last partial batch
        preds.extend(model.predict(img_array[ul:len(img_array)]))

    preds = np.array(preds)  # converting to a numpy array for easier handling.

    if individual_arrays is True:
        # 'disk_prob','unclass_prob','bulge_prob'
        return preds[:, 0], preds[:, 1], preds[:, 2]
    else:
        return preds


def gamornet_train_tflearn(training_imgs, training_labels, validation_imgs, validation_labels, input_shape, files_save_path="./", epochs=100, max_checkpoints=1, batch_size=64, lr=0.0001, momentum=0.9, decay=0.0, nesterov=False, loss='categorical_crossentropy', load_model=False, model_load_path="./", save_model=True, show_metric=True, clear_session=False):

    # TFLearn Loads graphs from memory by name, hence it's always advisable to set this to True if using in a Notebook.
    if clear_session is True:
        K.clear_session()

    check_imgs_validity(training_imgs)
    check_imgs_validity(validation_imgs)
    check_labels_validity(training_labels)
    check_labels_validity(validation_labels)

    model = gamornet_build_model_tflearn(input_shape=input_shape)

    if nesterov is False:
        optimizer = Momentum(momentum=momentum, lr_decay=decay)
    else:
        optimizer = Nesterov(momentum=momentum, lr_decay=decay)

    model = regression(model, optimizer=optimizer, loss=loss, learning_rate=lr)

    model = tflearn.DNN(model, checkpoint_path=files_save_path +
                        "check-", max_checkpoints=max_checkpoints)

    if load_model is True:
        model = gamornet_load_model_tflearn(model, model_load_path)

    model.fit(training_imgs, training_labels, n_epoch=epochs, validation_set=(validation_imgs, validation_labels),
              shuffle=True, show_metric=show_metric, batch_size=batch_size, snapshot_step=None, snapshot_epoch=save_model)

    return model


def gamornet_tl_tflearn(training_imgs, training_labels, validation_imgs, validation_labels, input_shape, load_layers_bools=[True]*8, trainable_bools=[True]*8, model_load_path="./", files_save_path="./", epochs=100, max_checkpoints=1, batch_size=64, lr=0.00001, momentum=0.9, decay=0.0, nesterov=False, loss='categorical_crossentropy', save_model=True, show_metric=True, clear_session=False):

    # TFLearn Loads graphs from memory by name, hence it's always advisable to set this to True if using in a Notebook.
    if clear_session is True:
        K.clear_session()

    check_imgs_validity(training_imgs)
    check_imgs_validity(validation_imgs)
    check_labels_validity(training_labels)
    check_labels_validity(validation_labels)

    model = gamornet_build_model_tflearn(
        input_shape=input_shape, trainable_bools=trainable_bools, load_layers_bools=load_layers_bools)

    if nesterov is False:
        optimizer = Momentum(momentum=momentum, lr_decay=decay)
    else:
        optimizer = Nesterov(momentum=momentum, lr_decay=decay)

    model = regression(model, optimizer=optimizer, loss=loss, learning_rate=lr)

    model = tflearn.DNN(model, checkpoint_path=files_save_path +
                        "check-", max_checkpoints=max_checkpoints)

    model = gamornet_load_model_tflearn(model, model_load_path)

    model.fit(training_imgs, training_labels, n_epoch=epochs, validation_set=(validation_imgs, validation_labels),
              shuffle=True, show_metric=show_metric, batch_size=batch_size, snapshot_step=None, snapshot_epoch=save_model)

    return model


###########################################
###########################################
