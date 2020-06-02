.. GaMorNet documentation master file, created by
   sphinx-quickstart on Tue May  5 19:36:48 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

About GaMorNet
==============
The Galaxy Morphology Network (GaMorNet) is a convolutional neural network which can classify galaxies as being disk-dominated, bulge-dominated or indeterminate based on their bulge to total light ratio. For more details about GaMorNet's design, how it was trained etc., please refer to :ref:`pub_and_other_data`.  

.. _first_contat:

First contact with GaMorNet
---------------------------
GaMorNet's user-faced functions has been written in a way so that it's easy to start using it even if you have not dealt with convolutional neural networks before. For. eg. to perform predictions on an array of SDSS images using our trained models, the following line of code is all you need. 

.. code-block:: python

   from gamornet.keras_module import gamornet_predict_keras

   preds = gamornet_predict_keras(img_array, model_load_path='SDSS_tl', input_shape='SDSS')


In order to get started with using GaMorNet, please first look at (getting started, tutorials followed by usage guide)



.. _pub_and_other_data:

Publication & Other Data
------------------------
You can look at this `ApJ paper <https://doi.org/10.3847/1538-4357/ab8a47>`_ to learn the details about GaMorNet's architecture, how it was trained, and other details not mentioned in this documentation. 

We strongly suggest you read the above-mentioned publication if you are going to use our trained models for performing predictions or as the starting point for your training. 

Attribution Info.
^^^^^^^^^^^^^^^^^^^
Please cite the above mentioned publication if you make use of this software module or some code herein.

.. code-block:: tex

    @article{Ghosh2020,
      doi = {10.3847/1538-4357/ab8a47},
      url = {https://doi.org/10.3847/1538-4357/ab8a47},
      year = {2020},
      month = jun,
      publisher = {American Astronomical Society},
      volume = {895},
      number = {2},
      pages = {112},
      author = {Aritra Ghosh and C. Megan Urry and Zhengdong Wang and Kevin Schawinski and Dennis Turp and Meredith C. Powell},
      title = {Galaxy Morphology Network: A Convolutional Neural Network Used to Study Morphology and Quenching in $\sim$100, 000 {SDSS} and $\sim$20, 000 {CANDELS} Galaxies},
      journal = {The Astrophysical Journal}
    }


Links to Additional Resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(links and instructions to trained models)


License
^^^^^^^^
Copyright 2020 Aritra Ghosh & contributors

Developed by `Aritra Ghosh <http://ghosharitra.com>`_ and made available under a `GNU GPL v3.0 <https://github.com/aritraghsh09/GaMorNet/blob/master/LICENSE>`_ license. 



.. _getting_help:

Getting Help
------------
Please first have a look at the ..


.. toctree::
   :maxdepth: 2
   :caption: Documentation  Contents:

   getting_started
   tutorials
   usage_guide
   api_docs
   faq

