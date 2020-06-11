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


def gamornet_predict_tflearn(img_array, model_load_path, input_shape, batch_size=64, individual_arrays=False, trainable_bools=[True]*8, 
                             clear_session=False):

    """
    Uses a `tflearn` model to perform predictions on supplied images. 

    Parameters
    ----------
    img_array: Numpy ndarray[nsamples, x, y, ndim]
        The array of images on which the predictions are to be performed. We insist on numpy arrays as many of the 
        underlying deep learning frameworks work better with numpy arrays compared to other array-like elements. 

    model_load_path: str 
        Path to the saved model. Note that tflearn models are usually consist of three files in the format file_name.``data``,
        file_name.``info``, file_name.``meta``. For this parameter, simply specify file_path/file_name.
        
        This parameter can take the following special values

        * ``SDSS_sim`` -- Downloads and uses GaMorNet models trained on SDSS g-band simulations a z~0 from Ghosh et. al. (2020)
        * ``SDSS_tl`` -- Downloads and uses GaMorNet models trained on SDSS g-band simulations and real data at z~0 from Ghosh et. al. (2020)
        * ``CANDELS_sim`` -- Downloads and uses GaMorNet models trained on CANDELS H-band simulations a z~1 from Ghosh et. al. (2020)
        * ``CANDELS_tl`` -- Downloads and uses GaMorNet models trained on CANDELS H-band simulations and real data at z~1 from Ghosh et. al. (2020)

    input_shape: tuple of ints (x, y, ndim) or allowed str
        The shape of the images being used. The parameter can also take the following special values:-

        * ``SDSS`` - Sets the input shape to be (167,167,1) as was used for the SDSS g-band images in Ghosh et. al. (2020)
        * ``CANDELS`` -  Sets the input shape to be (83,83,1) as was used for the CANDELS H-band images in Ghosh et. al. (2020)

    batch_size: int
        This variable specifies how many images will be processed in a single batch. Set this value to lower than the default
        if you have limited memory availability. This doesn't affect the predictions in any way. 

    individual_arrays: bool
        If set to True, this will unpack the three returned arrays 

    trainable_bools: array of bools or allowed str
        This variable is used to identify which of the 5 convolutional and 3 fully-connected layers of GaMorNet were 
        set to trainable during the training phase of the model (which is now being used for prediction)

        The orders of the bools correspond to the Following Layer numbers [2,5,8,9,10,13,15,17] in GaMorNet. Please see 
        Figure 4 and Table 2 of Ghosh et. al. (2020) to get more details The first five layers are the convolutional
        layers and the last three are the fully connected layers.  

        This parameter can also take the following special values which are handy when you are using our models to
        perform predictions:-

        * ``train_bools_SDSS`` - Sets the bools according to what was done for the SDSS data in Ghosh et. al. (2020)
        * ``train_bools_CANDELS``- Sets the bools according to what was done for the CANDELS data in Ghosh et. al. (2020)

    clear_session: bool
        If set to True, this will clear the TensorFlow session currently running. This is handy while running GaMorNet in a 
        notebook to avoid variable name confusions. (Sometimes, under the hood, TFLearn & Tensorflow reuses the same layer names 
        leading to conflicts)

        Note that, if set to True, you will lose access to any other graphs you may have run before. 


    Returns
    -------
    predicted probabilities: array_like
        The returned array consists of the probability for each galaxy to be disk-dominated, indeterminate and bulge-dominated 
        respectively [disk_prob,indet_prob,bulge_prob].If individual arrays are set to True, the single array is unpacked 
        and returned  as three separate arrays in the same order. 

        The ordering of individual elements in this array corresponds to the array of images fed in. 


    """

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
    time.sleep(0.3)


    for i in progressbar.progressbar(range(0, num_batches)):
        ll = i*batch_size
        ul = (i+1)*batch_size
        preds.extend(model.predict(img_array[ll:ul]))

    
    if num_batches == 0: #when batch_size > number of images
        preds.extend(model.predict(img_array[0:len(img_array)]))

    elif ul != len(img_array):# for the last partial batch
        preds.extend(model.predict(img_array[ul:len(img_array)]))

    preds = np.array(preds)  # converting to a numpy array for easier handling.

    if individual_arrays is True:
        # 'disk_prob','unclass_prob','bulge_prob'
        return preds[:, 0], preds[:, 1], preds[:, 2]
    else:
        return preds


def gamornet_train_tflearn(training_imgs, training_labels, validation_imgs, validation_labels, input_shape, files_save_path="./", epochs=100, 
                           max_checkpoints=1, batch_size=64, lr=0.0001, momentum=0.9, decay=0.0, nesterov=False, loss='categorical_crossentropy', 
                           load_model=False, model_load_path="./", save_model=True, show_metric=True, clear_session=False):


    """
    Trains and return a GaMorNet model using TFLearn. 

    Parameters
    -----------

    training_imgs: Numpy ndarray [nsamples,x,y,ndim]
        The array of images on which are to be used for the training process. We insist on numpy arrays 
        as many of the underlying deep learning frameworks work better with numpy arrays compared to 
        other array-like elements.

    training_labels: Numpy ndarray [nsamples,label_arrays]
        The truth labels for each of the training images. The supplied labels must be in the one-hot encoding 
        format. We reproduce below what each individual label array should look like:-

        * Disk-dominated - ``[1,0,0]``
        * Indeterminate -  ``[0,1,0]``
        * Bulge-dominated - ``[0,0,1]``

    validation_imgs: Numpy ndarray [nsamples,x,y,ndim]
        The array of images on which are to be used for the validation process. We insist on numpy arrays 
        as many of the underlying deep learning frameworks work better with numpy arrays compared to 
        other array-like elements.

    validation_labels: Numpy ndarray [nsamples,label_arrays]
        The truth labels for each of the validation images. The supplied labels must be in the one-hot encoding 
        format. We reproduce below what each individual label array should look like:-

        * Disk-dominated - ``[1,0,0]``
        * Indeterminate -  ``[0,1,0]``
        * Bulge-dominated - ``[0,0,1]``

    input_shape: tuple of ints (x, y, ndim) or allowed str
        The shape of the images being used. The parameter can also take the following special values:-

        * ``SDSS`` - Sets the input shape to be (167,167,1) as was used for the SDSS g-band images in Ghosh et. al. (2020)
        * ``CANDELS`` -  Sets the input shape to be (83,83,1) as was used for the CANDELS H-band images in Ghosh et. al. (2020)

    files_save_path: str
        The full path to the location where the model generated during the training process are to be 
        saved. The path should end with the name of the file. For eg. ``/path/checkpoint``. This
        will result in model files of the form ``checkpoint.meta", ``checkpoint.data`` and
        ``checkpoint.info`` being saved. 

        Set this to `/dev/null` on a unix system if you don't want to save the file(s)

    epochs: int
        The number of epochs for which you want to training the model. 

    max_checkpoints: int
        TFLearn saves the model at the end of each epoch. This parameter controls how many of the 
        most recent models are saved. For eg. setting this to 2, will save the model state during the 
        most recent two epochs. 

    batch_size: int
        This variable specifies how many images will be processed in a single batch. This is a 
        hyperparameter. The default value is a good starting point

    lr: float
        This is the learning rate to be used during the training process. This is a 
        hyperparameter that should be tuned during the training process. The default value is a good
        starting point.

    momentum: float
        The value momentum to be used in the gradient descent optimizer that is used to train GaMorNet. 
        This must always be :math:`\geq 0`. This accelerates the gradient descent process. This is a 
        hyperparameter. The default value is a good starting point. 

    decay: float
        The amount of learning rate decay to be applied over each update. 

    nesterov: bool
        Whether to apply Nesterov momentum or not. 

    loss: allowed str or function
        The loss function to be used. If using the string option, you need to supply the name of 
        the loss function. This can be set to be any loss available in ``tflearn``

    load_model: bool
        Whether you want to start the training from a previously saved model. 

        We strongly recommend using the ``gamornet_tl_keras`` function for more 
        control over the process when starting the training from a previously
        saved model.

    model_load_path: str
        Required `iff load_model ==True`. The path to the saved model.

        Note that tflearn models are usually consist of three files in the format 
        file_name.``data``, file_name.``info``, file_name.``meta``. For this parameter,
        simply specify file_path/file_name.

    save_model: bool
        Whether you want to save the model files at each epoch during training. This
        parameter should be used in conjunction with  ``max_checkpoints`` to configure
        how many of the saved model files are preserved till the end. 

    show_metric: bool
        Whether to display the training/testing metrics during training.

    clear_session: bool
        If set to True, this will clear the TensorFlow session currently running. This is handy while running GaMorNet in a 
        notebook to avoid variable name confusions. (Sometimes, under the hood, TFLearn & Tensorflow reuses the same layer names 
        leading to conflicts)

        Note that, if set to True, you will lose access to any other graphs you may have run before. 


    Returns
    --------

    Trained TFLearn Model: TFLearn ``models.dnn.DNN`` class

    """


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
                        "check", max_checkpoints=max_checkpoints)

    if load_model is True:
        model = gamornet_load_model_tflearn(model, model_load_path)

    model.fit(training_imgs, training_labels, n_epoch=epochs, validation_set=(validation_imgs, validation_labels),
              shuffle=True, show_metric=show_metric, batch_size=batch_size, snapshot_step=None, snapshot_epoch=save_model)

    return model


def gamornet_tl_tflearn(training_imgs, training_labels, validation_imgs, validation_labels, input_shape, load_layers_bools=[True]*8, 
                        trainable_bools=[True]*8, model_load_path="./", files_save_path="./", epochs=100, max_checkpoints=1, batch_size=64, 
                        lr=0.00001, momentum=0.9, decay=0.0, nesterov=False, loss='categorical_crossentropy', save_model=True, 
                        show_metric=True, clear_session=False):

    """
    Performs Transfer Learning (TL) using a previously trained GaMorNet model. 

    Parameters
    -----------

    training_imgs: Numpy ndarray [nsamples,x,y,ndim]
        The array of images on which are to be used for the TL process. We insist on numpy arrays 
        as many of the underlying deep learning frameworks work better with numpy arrays compared to 
        other array-like elements.

    training_labels: Numpy ndarray [nsamples,label_arrays]
        The truth labels for each of the TL images. The supplied labels must be in the one-hot encoding 
        format. We reproduce below what each individual label array should look like:-

        * Disk-dominated - ``[1,0,0]``
        * Indeterminate -  ``[0,1,0]``
        * Bulge-dominated - ``[0,0,1]``

    validation_imgs: Numpy ndarray [nsamples,x,y,ndim]
        The array of images on which are to be used for the validation process. We insist on numpy arrays 
        as many of the underlying deep learning frameworks work better with numpy arrays compared to 
        other array-like elements.

    validation_labels: Numpy ndarray [nsamples,label_arrays]
        The truth labels for each of the validation images. The supplied labels must be in the one-hot encoding 
        format. We reproduce below what each individual label array should look like:-

        * Disk-dominated - ``[1,0,0]``
        * Indeterminate -  ``[0,1,0]``
        * Bulge-dominated - ``[0,0,1]``

    input_shape: tuple of ints (x, y, ndim) or allowed str
        The shape of the images being used. The parameter can also take the following special values:-

        * ``SDSS`` - Sets the input shape to be (167,167,1) as was used for the SDSS g-band images in Ghosh et. al. (2020)
        * ``CANDELS`` -  Sets the input shape to be (83,83,1) as was used for the CANDELS H-band images in Ghosh et. al. (2020)

    load_layers_bools: array of bools
        This variable is used to identify which of the 5 convolutional and 3 fully-connected layers of GaMorNet will be 
        loaded during the transfer learning process from the supplied starting model. The rest of the layers will be
        initialized from scratch.

        The orders of the bools correspond to the Following Layer numbers [2,5,8,9,10,13,15,17] in GaMorNet. Please see 
        Figure 4 and Table 2 of Ghosh et. al. (2020) to get more details The first five layers are the convolutional
        layers and the last three are the fully connected layers.  

        This parameter can also take the following special values which are handy when you are using our models to
        perform predictions:-

        * ``load_bools_SDSS`` - Sets the bools according to what was done for the SDSS data in Ghosh et. al. (2020)
        * ``load_bools_CANDELS``- Sets the bools according to what was done for the CANDELS data in Ghosh et. al. (2020)

    trainable_bools: array of bools
        This variable is used to identify which of the 5 convolutional and 3 fully-connected layers of GaMorNet will be 
        trainable during the transfer learning process. The rest are frozen at the values loaded from the previous
        model.

        The orders of the bools correspond to the Following Layer numbers [2,5,8,9,10,13,15,17] in GaMorNet. Please see 
        Figure 4 and Table 2 of Ghosh et. al. (2020) to get more details The first five layers are the convolutional
        layers and the last three are the fully connected layers.  

        This parameter can also take the following special values which are handy when you are using our models to
        perform predictions:-

        * ``train_bools_SDSS`` - Sets the bools according to what was done for the SDSS data in Ghosh et. al. (2020)
        * ``train_bools_CANDELS``- Sets the bools according to what was done for the CANDELS data in Ghosh et. al. (2020)

    model_load_path: str
        Path to the saved model, which will serve as the starting point for transfer learning. Note that 
        tflearn models are usually consist of three files in the format file_name.``data``,
        file_name.``info``, file_name.``meta``. For this parameter, simply specify file_path/file_name.

        This parameter can also take the following special values
        
        * ``SDSS_sim`` -- Downloads and uses GaMorNet models trained on SDSS g-band simulations a z~0 from Ghosh et. al. (2020)
        * ``SDSS_tl`` -- Downloads and uses GaMorNet models trained on SDSS g-band simulations and real data at z~0 from Ghosh et. al. (2020)
        * ``CANDELS_sim`` -- Downloads and uses GaMorNet models trained on CANDELS H-band simulations a z~1 from Ghosh et. al. (2020)
        * ``CANDELS_tl`` -- Downloads and uses GaMorNet models trained on CANDELS H-band simulations and real data at z~1 from Ghosh et. al. (2020)

    files_save_path: str
        The full path to the location where the model generated during the training process are to be 
        saved. The path should end with the name of the file. For eg. ``/path/checkpoint``. This
        will result in model files of the form ``checkpoint.meta", ``checkpoint.data`` and
        ``checkpoint.info`` being saved. 

        Set this to `/dev/null` on a unix system if you don't want to save the output. 

    epochs: int
        The number of epochs for which you want to training the model. 

    max_checkpoints: int
        TFLearn saves the model at the end of each epoch. This parameter controls how many of the 
        most recent models are saved. For eg. setting this to 2, will save the model state during the 
        most recent two epochs.

    batch_size: int
        This variable specifies how many images will be processed in a single batch. This is a 
        hyperparameter. The default value is a good starting point

    lr: float
        This is the learning rate to be used during the training process. This is a 
        hyperparameter that should be tuned during the training process. The default value is a good
        starting point.

    momentum: float
        The value momentum to be used in the gradient descent optimizer that is used to train GaMorNet. 
        This must always be :math:`\geq 0`. This accelerates the gradient descent process. This is a 
        hyperparameter. The default value is a good starting point. 

    decay: float
        The amount of learning rate decay to be applied over each update. 

    nesterov: bool
        Whether to apply Nesterov momentum or not. 

    loss: allowed str or function
        The loss function to be used. If using the string option, you need to supply the name of 
        the loss function. This can be set to be any loss available in ``tflearn``

    save_model: bool
        Whether you want to save the model files at each epoch during training. This
        parameter should be used in conjunction with  ``max_checkpoints`` to configure
        how many of the saved model files are preserved till the end. 

    show_metric: bool
        Whether to display the training/testing metrics during training.

    clear_session: bool
        If set to True, this will clear the TensorFlow session currently running. This is handy while running GaMorNet in a 
        notebook to avoid variable name confusions. (Sometimes, under the hood, TFLearn & Tensorflow reuses the same layer names 
        leading to conflicts)

        Note that, if set to True, you will lose access to any other graphs you may have run before. 


    Returns
    --------

    Trained TFLearn Model: TFLearn ``models.dnn.DNN`` class
        
    """

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
                        "check", max_checkpoints=max_checkpoints)

    model = gamornet_load_model_tflearn(model, model_load_path)

    model.fit(training_imgs, training_labels, n_epoch=epochs, validation_set=(validation_imgs, validation_labels),
              shuffle=True, show_metric=show_metric, batch_size=batch_size, snapshot_step=None, snapshot_epoch=save_model)

    return model


###########################################
###########################################
