.. _getting_started:

Getting Started
===============

GaMorNet is module written in Python and uses the `Keras <https://keras.io>`_ and `TFLearn <http://tflearn.org>`_ deep learning libraries to perform all of the machine learning operations. Both these aforementioned libraries in turn use `TensorFlow <https://www.tensorflow.org>`_ for their underlying tensor operations. GaMorNet was originally written using TFLearn, but the Keras module was added later on as we expect Keras to be better supported and developed going forward. 

Additionally, GaMorNet has two separate packages available on `pip <https://pypi.org>`_. One happens to be the standard ``gamornet`` package and the other one is a ``gamornet-cpu`` package meant for users who don't have access to a CPU. 

Ways to Use GaMorNet
--------------------

#. If you have access to a GPU,

        * We recommend installing the ``gamornet`` package using the instructions in :ref:`installation`


#. If you don't have access to a GPU, 
    
    * and want to use our models for predictions
    
        * You can install the ``gamornet-cpu`` package using the instructions in :ref:`installation`
        * You can use Google Colab like we have done in the :ref:`tutorials`.

    * and want to train your own models

        * Use the GPUs available via Google Colab as we have done in the :ref:`tutorials`



.. _installation:

Installation
-------------


Complatibility with CUDA & cuDNN
--------------------------------
