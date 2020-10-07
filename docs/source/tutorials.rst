.. _tutorials:

Tutorials
=========
.. error::
   **6th Oct. 2020**: Yale Astronomy's public FTP server is temporarily down. Yale
   ITS is working to restore the service as soon as possible and it should 
   be back online by Oct 17th. 

   During this time, tutorials that depend on pulling files from the server will
   not work as expected. Besides, in your code, if you are trying to use our 
   trained models, this will fail as well. We apologize for this inconvenience. If your work is urgent, please reach out
   to us and we can make the trained models available to you via Google Drive. 



We have created the following tutorials to get you quickly started with using GaMorNet. To look into the details of each GaMorNet function used in these tutorials, please look at the :ref:`api_docs`.

You can either download these notebooks from GitHub and run them on your own machine or use `Google Colab <https://colab.research.google.com/>`_ to run these using Google GPUs. 

Each Notebook has separate sections on using the Keras and TFLearn modules. 


.. _prediction_tutorial:

Making Predictions
------------------

This tutorial demonstrates how you can use GaMorNet models to make predictions using two images from our SDSS dataset. 

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/aritraghsh09/GaMorNet/blob/master/tutorials/gamornet_predict_tutorial.ipynb
    :alt: Run in Google Colab

.. image:: https://img.shields.io/badge/|%20-Open%20in%20GitHub-informational?logo=github
    :target: https://github.com/aritraghsh09/GaMorNet/blob/master/tutorials/gamornet_predict_tutorial.ipynb
    :alt: Open in GitHub


.. _training_tutorial:

Training GaMorNet
-----------------

This tutorial uses simulated SDSS galaxies to train a GaMorNet model from scratch. 

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/aritraghsh09/GaMorNet/blob/master/tutorials/gamornet_train_tutorial.ipynb
    :alt: Run in Google Colab

.. image:: https://img.shields.io/badge/|%20-Open%20in%20GitHub-informational?logo=github
    :target: https://github.com/aritraghsh09/GaMorNet/blob/master/tutorials/gamornet_train_tutorial.ipynb
    :alt: Open in GitHub


.. _tl_tutorial:

Transfer Learning with GaMorNet
-------------------------------

This tutorial uses real SDSS galaxies to perform transfer learning on a GaMorNet model trained only on simulations. 

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/aritraghsh09/GaMorNet/blob/master/tutorials/gamornet_tl_tutorial.ipynb
    :alt: Run in Google Colab

.. image:: https://img.shields.io/badge/|%20-Open%20in%20GitHub-informational?logo=github
    :target: https://github.com/aritraghsh09/GaMorNet/blob/master/tutorials/gamornet_tl_tutorial.ipynb
    :alt: Open in GitHub