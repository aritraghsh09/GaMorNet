
.. image:: https://readthedocs.org/projects/gamornet/badge/?version=latest
    :target: https://gamornet.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/license-GPL%20v3.0-blue
    :target: https://github.com/aritraghsh09/GaMorNet/blob/master/LICENSE
    :alt: License Information

.. image:: https://badge.fury.io/py/gamornet.svg
    :target: https://pypi.org/project/gamornet/
    :alt: PyPI version

.. image:: https://img.shields.io/badge/doi-10.3847%2F1538--4357%2Fab8a47-blue
    :target: https://doi.org/10.3847/1538-4357/ab8a47
    :alt: DOI Link

.. image:: https://img.shields.io/badge/arXiv-2006.14639-blue
    :target: http://arxiv.org/abs/2006.14639
    :alt: arXiv


Galaxy Morphology Network (GaMorNet)
=====================================

GaMorNet is a Convolutional Neural Network based on AlexNet to classify galaxies morphologically. GaMorNet does not need a large amount of training data (as it is trained on simulations and then transfer-learned on a small portion of real data) and can be applied on multiple datasets. Till now, GaMorNet has been tested on ~100,000 SDSS g-band galaxies and ~20,000 CANDELS H-band galaxies and has a misclassification rate of <5%. 


Documentation
-------------

Please read the `detailed documentation <https://gamornet.readthedocs.io/>`_ in order to start using GaMorNet


Publication & Other Data
------------------------
You can look at this `ApJ paper <https://doi.org/10.3847/1538-4357/ab8a47>`_ to learn the details about GaMorNet's architecture, how it was trained, and other details not mentioned in the documentation. 

We strongly suggest you read the above-mentioned publication if you are going to use our trained models for performing predictions or as the starting point for training your own models.

All the different elements of the public data release (including the new Keras models) are summarized in the `PDR Usage Guide <https://gamornet.readthedocs.io/en/latest/usage_guide.html>`_


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

Additionally, if you want, please include the following text in the Software/Acknowledgment section.

.. code-block:: tex

    This work uses trained models/software made available as a part of the Galaxy Morphology Network public data release. 


License
^^^^^^^^
Copyright 2020 Aritra Ghosh & contributors

Developed by `Aritra Ghosh <http://ghosharitra.com>`_ and made available under a `GNU GPL v3.0 <https://github.com/aritraghsh09/GaMorNet/blob/master/LICENSE>`_ license. 



.. _getting_help:

Getting Help/Contributing
--------------------------
If you have a question, please first have a look at the `FAQ Section <https://gamornet.readthedocs.io/en/latest/faq.html>`_ . If your question is not answered there, please send me an e-mail at this ``aritraghsh09+gamornet@xxxxx.com`` GMail address.

If you have spotted a bug in the code/documentation or you want to propose a new feature, please feel free to open an issue/a pull request on `GitHub <https://github.com/aritraghsh09/GaMorNet>`_ .


