.. _api_docs:

API Documentation
=================

Both the Keras and TFLearn modules have similarly named functions with very similar parameters. Use the ``_predict_`` functions to perform
predictions using our trained models or a model you trained from scratch. Use the ``_train_`` functions to train a model from scratch. Use the 
``_tl_`` functions to perform transfer learning on a previously trained model -- this can be our pre-trained models or a model that you trained.

Please have a look at the :ref:`tutorials` for examples of how to use these functions effectively.

Keras Module
-------------

.. module:: gamornet.keras_module

The three major user oriented functions happen to be :func:`gamornet_predict_keras`, :func:`gamornet_train_keras` and :func:`gamornet_tl_keras` 
and are documented here. For the remainder of the functions, please have a look at the source code on GitHub. 


.. autofunction:: gamornet.keras_module.gamornet_predict_keras


.. autofunction:: gamornet.keras_module.gamornet_train_keras


.. autofunction:: gamornet.keras_module.gamornet_tl_keras



TFLearn Module
--------------

.. module:: gamornet.tflearn_module

The three major user oriented functions happen to be :func:`gamornet_predict_tflearn`, :func:`gamornet_train_tflearn` and :func:`gamornet_tl_tflearn`
and are documented here. For the remainder of the functions, please have a look at the source code on GitHub. 


.. autofunction:: gamornet.tflearn_module.gamornet_predict_tflearn


.. autofunction:: gamornet.tflearn_module.gamornet_train_tflearn


.. autofunction:: gamornet.tflearn_module.gamornet_tl_tflearn
