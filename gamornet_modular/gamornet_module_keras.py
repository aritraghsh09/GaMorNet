from keras.models import Sequential, load_model, clone_model
from keras.layers import Dense, Flatten, Dropout, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import VarianceScaling
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
import pandas as pd
import wget
import random
import string
import os


#Implementing LRN in Keras. Code from "Deep Learning with Python" by F. Chollet
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
        if K.image_dim_ordering == "th":
            _, f, r, c = self.shape
        else:
            _, r, c, f = self.shape
        squared = K.square(x)
        pooled = K.pool2d(squared, (self.n, self.n), strides=(1, 1),
            padding="same", pool_mode="avg")
        if K.image_dim_ordering == "th":
            summed = K.sum(pooled, axis=1, keepdims=True)
            averaged = self.alpha * K.repeat_elements(summed, f, axis=1)
        else:
            summed = K.sum(pooled, axis=3, keepdims=True)
            averaged = self.alpha * K.repeat_elements(summed, f, axis=3)
        denom = K.pow(self.k + averaged, self.beta)
        return x / denom

    def get_output_shape_for(self, input_shape):
        return input_shape
    

def check_input_shape_validity(input_shape):
    
    if input_shape == 'SDSS':
        input_shape = (167,167,1)
    elif input_shape == 'CANDELS':
        input_shape = (83,83,1)
        
    return input_shape
    

def gamornet_build_model_keras(input_shape):
    
    input_shape = check_input_shape_validity(input_shape)
    
    #uniform scaling initializer
    uniform_scaling = VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None)

    #Building GaMorNet
    model = Sequential()

    model.add(Conv2D(96,11, strides=4, activation='relu', input_shape=input_shape,padding='same',kernel_initializer=uniform_scaling))
    model.add(MaxPooling2D(pool_size=3, strides=2,padding='same'))
    model.add(LocalResponseNormalization())

    model.add(Conv2D(256,5, activation='relu',padding='same',kernel_initializer=uniform_scaling))
    model.add(MaxPooling2D(pool_size=3, strides=2,padding='same'))
    model.add(LocalResponseNormalization())

    model.add(Conv2D(384, 3, activation='relu',padding='same',kernel_initializer=uniform_scaling))
    model.add(Conv2D(384, 3, activation='relu',padding='same',kernel_initializer=uniform_scaling))
    model.add(Conv2D(256, 3, activation='relu',padding='same',kernel_initializer=uniform_scaling))
    model.add(MaxPooling2D(pool_size=3, strides=2,padding='same'))
    model.add(LocalResponseNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='tanh',kernel_initializer='TruncatedNormal'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='tanh',kernel_initializer='TruncatedNormal'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax',kernel_initializer='TruncatedNormal'))
    
    return model

def get_model_from_link_keras(link,model):
    
    letters = string.ascii_lowercase
    if link[-4:] == 'hdf5':
        file_name = ''.join(random.choice(letters) for i in range(15)) + '.hdf5'
    else:
        file_name = ''.join(random.choice(letters) for i in range(15)) + '.h5'
    
    wget.download(link,out=file_name)
    model.load_weights(file_name)
    os.remove(file_name)
    
    return model

def gamornet_load_model_keras(model,model_load_path):
    
    if model_load_path == 'SDSS_sim':
        print("Fetching SDSS Sim Trained Weigths.....")
        link = 'ftp://ftp.astro.yale.edu/pub/aghosh/gamornet_keras/trained_models/SDSS/sim_trained/model_sdsss_sim_trained.h5'
        model = get_model_from_link_keras(link,model)
        
    elif model_load_path == 'SDSS_tl':
        print("Fetching SDSS TL Weigths.....")
        link = 'ftp://ftp.astro.yale.edu/pub/aghosh/gamornet_keras/trained_models/SDSS/tl/model_sdsss_tl.hdf5'
        model = get_model_from_link_keras(link,model)
        
    elif model_load_path == 'CANDELS_sim':
        print("Fetching CANDELS Sim Trained Weigths.....")
        link = 'ftp://ftp.astro.yale.edu/pub/aghosh/gamornet_keras/trained_models/CANDELS/sim_trained/model_candels_sim_trained.hdf5'
        model = get_model_from_link_keras(link,model)
        
    elif model_load_path == 'CANDELS_tl':
        print("Fetching CANDELS TL Weigths.....")
        link = 'ftp://ftp.astro.yale.edu/pub/aghosh/gamornet_keras/trained_models/CANDELS/tl/model_candels_tl.hdf5'
        model = get_model_from_link_keras(link,model)
        
    else:
         model.load_weights(model_load_path)
            
    return model
    


def gamornet_predict_keras(img_array,model_load_path,input_shape,batch_size=64,individual_arrays=False):
    
    model = gamornet_build_model_keras(input_shape=input_shape)
    
    print("Loading GaMorNet Model.....")
    model = gamornet_load_model_keras(model,model_load_path)
    
    print("Performing Predictions.....")
    preds = model.predict(img_array,batch_size = batch_size)
    preds = np.array(preds) #converting to a numpy array for easier handling.
    
    if individual_arrays == True:
        return preds[:,0],preds[:,1],preds[:,2] # 'disk_prob','unclass_prob','bulge_prob'
    else:
        return preds
