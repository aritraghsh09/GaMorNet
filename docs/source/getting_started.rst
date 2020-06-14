.. _getting_started:

Getting Started
===============

GaMorNet is written in Python and uses the `Keras <https://keras.io>`_ and `TFLearn <http://tflearn.org>`_ deep learning libraries to perform all of the machine learning operations. Both these aforementioned libraries in turn use `TensorFlow <https://www.tensorflow.org>`_ for their underlying tensor operations. GaMorNet was originally written using TFLearn, but the Keras module was added later as we expect Keras to be better supported and developed going forward. 

GaMorNet has two separate packages available via `pip <https://pypi.org>`_. One happens to be the standard ``gamornet`` package and the other one is a ``gamornet-cpu`` package meant for users who don't have access to a GPU. 

Ways to Use GaMorNet
--------------------

#. If you have access to a GPU,

        * We recommend installing the ``gamornet`` package using the instructions in :ref:`installation`
        * However, if you are not familiar with how to enable GPU support for TensorFlow and want to get started quickly, you may consider using Google Colab like we have done in the :ref:`tutorials`


#. If you don't have access to a GPU, 
    
    * and want to use our models for predictions
    
        * You can install the ``gamornet-cpu`` package using the instructions in :ref:`installation`
        * You can use Google Colab like we have done in the :ref:`tutorials`

    * and want to train your own models

        * Use the GPUs available via Google Colab as we have done in the :ref:`tutorials`



.. _installation:

Installation
-------------

It is highly recommended to have a separate Python `virtual environment <https://medium.com/@pinareceaktan/what-is-this-virtual-environments-in-python-and-why-anyone-ever-needs-them-7e3e682f9d2>`_ for running GaMorNet as the package has many specific version oriented dependencies on other Python packages. The following instructions are shown using `Anaconda <https://www.anaconda.com/products/individual>`_, but feel free to go ahead and use any other virtual environment tool you are comfortable using. **Note that GaMorNet only runs on Python >= 3.3 and is recommended to be run on Python 3.6**

1. Using pip

    * Install Anaconda if you don't have it already using the instructions `here <https://www.anaconda.com/products/individual>`_

    * Create a new Anaconda environment using ``conda create -n gamornetenv python=3.6``

    * Activate the above environment using ``conda activate gamornetenv``

    * Install GaMorNet using ``pip install gamornet`` or ``pip install gamornet-cpu`` depending on your requirements

    * For the GPU installation, if you don't have the proper CUDA libraries, please see :ref:`compatibility`

    * To test the installation, open up a Python shell and type ``from gamornet.keras_module import *``. If this doesn't raise any errors, it means you have installed GaMorNet successfully. 

    * To exit the virtual environment, type ``conda deactivate``



2. From Source

    * Install Anaconda if you don't have it already using the instructions `here <https://www.anaconda.com/products/individual>`_

    * Create a new Anaconda environment using ``conda create -n gamornetenv python=3.6``

    * Activate the above environment using ``conda activate gamornetenv``

    * Clone GaMorNet repository from GitHub using ``git clone https://github.com/aritraghsh09/GaMorNet.git``

    * To install, do the following based on the package you want

        * For GPU installation,

            * ``cd GaMorNet``
            * ``python setup.py install``

        * For CPU version,

            * ``cd GaMorNet``
            * ``git fetch --all``
            * ``git checkout cpu_version``
            * ``python setup.py install``

    * For the GPU installation, if you don't have the proper CUDA libraries, please see :ref:`compatibility`

    * To test the installation, open up a Python shell and type ``from gamornet.keras_module import *``. If this doesn't raise any errors, it means you have installed GaMorNet successfully. 

    * To exit the virtual environment, type ``conda deactivate``


.. _compatibility:

GPU Support
------------

If you are using a GPU, then you would need to make sure that the appropriate CUDA and cuDNN versions are installed. The appropriate version is decided by the versions of your installed Python libraries. For detailed instructions on how to enable GPU support for Tensorflow, please see this `link <https://www.tensorflow.org/install/gpu>`_. 

We tested GaMorNet using the following configurations:-

===========  =========  ========== ========== ========== ==========
Python       Keras      TFLearn    Tensorflow CUDA       cuDNN
===========  =========  ========== ========== ========== ==========
3.6.10       2.2.4      0.3.2      1.13.1     10.0.130   7.6.0
3.6.10       2.3.1      0.3.2      1.15.3     10.0.130   7.6.2
===========  =========  ========== ========== ========== ==========

For more build configurations tested out by the folks at TensorFlow, please see `this link <https://www.tensorflow.org/install/source#linux>`_



