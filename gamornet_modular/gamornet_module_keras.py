from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import VarianceScaling
from keras import backend as K
from keras.engine.topology import Layer, InputSpec

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
    
    

def build_gamornet_model_keras(input_shape=None):
    
    if input_shape == None:
        print("ERROR:You need to specify an input shape while calling build_gamornet_model")
    elif input_shape == 'SDSS':
        input_shape = (167,167,1)
    elif input_shape == 'CANDELS':
        input_shape = (83,83,1)
    
    
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
