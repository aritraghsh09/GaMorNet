.. _faq:

FAQs
====


#. Can I run GaMorNet on any galaxy image?

    No! Please see our recommendations in the :ref:`usage_guide`.

#. I am having difficulty enabling GPU support. What should I do?

    Try using Google Colab like we have done in the :ref:`tutorials`. 

    Note that the underlying package that we use to interact with a GPU is TensorFlow. Look at `these <https://www.tensorflow.org/install/gpu>`_ detailed instructions for enabling GPU support for TensorFlow. Alternatively, if you are running this on a supercomputer, ask the administrators for detailed instructions on installing TensorFlow. 


#. I am getting an import error involving ``GLIBC`` or ``libcudas.so``  or ``libm.so``.

    In all probability, you are getting these errors because TensorFlow cannot find the appropriate CUDA libraries. Please follow the instructions `here <https://www.tensorflow.org/install/gpu>`_. Alternatively if you are running this on a supercomputer, ask the administrators for detailed instructions on installing TensorFlow. 


#. Should I use the Keras or TFLearn module if I myself don't have a preference?

    We recommend using the Keras module as we expect it to be better supported going forward. However, you may wish to take a look at the :ref:`usage_guide` for differences between the two modules. It should be noted that the results in the original paper was obtained using TFLearn. 

#. Is it worth enabling GPU support?

    We highly recommend running GaMorNet on a GPU if you are going to train your own models. 


#. What if my question is not answered here?

    Please send me an e-mail at this ``aritraghsh09+gamornet@xxxxx.com`` GMail address. Additionally, if you have spotted a bug in the code/documentation or you want to propose a new feature, please feel free to open an issue/a pull request on `GitHub <https://github.com/aritraghsh09/GaMorNet>`_
