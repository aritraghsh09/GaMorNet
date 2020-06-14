.. _usage_guide:

Public Data Release Handbook
=============================

If you are looking for information about the various ways you can use GaMorNet (running on a CPU v/s GPU v/s the cloud) or installation instructions, please have a look at :ref:`getting_started`. This section summarizes different aspects of the public data release and provides some advice on the applicability of GaMorNet for various tasks. 


.. _usage_advice:

Usage Advice
-------------
How you will use the public data release of GaMorNet strongly depends on the task at hand. 

* If you are looking for predictions of the SDSS g-band and CANDELS H-band dataset of |Ghosh et. al. (2020)|_, please have a look at the :ref:`pred_tables` section. 

* If you have SDSS g-band (:math:`z \sim 0`) and/or CANDELS H-band (:math:`z \sim 1`) data that we haven't classified, please use the final trained models (on simulations + real data) that we have released. You can manually download these models from :ref:`trained_models` or use the :func:`gamornet_predict_keras` / :func:`gamornet_predict_tflearn` functions as shown in :ref:`tutorials` and :ref:`api_docs`. 

* If you have SDSS and CANDELS data other than g-band at :math:`z \sim 0` and H-band at :math:`z \sim 1` that you want to classify:- 

    * If the data are in nearby bands *at the same redshifts* (i.e. near g-band for SDSS and H-band for CANDELS), we recommend using the :func:`gamornet_tl_keras` / :func:`gamornet_tl_tflearn` functions as shown in :ref:`tutorials` and :ref:`api_docs` to perform transfer learning. We recommend starting the transfer learning process from both our simulation-only and final trained models and choosing the one that maximizes the accuracy on your validation set. In case you want to download the models manually, see :ref:`trained_models`.

    * If you believe that your data is significantly different in redshift, resolution or any other photometric aspect, you could also train a network from scratch using :func:`gamornet_train_keras` / :func:`gamornet_train_tflearn` as shown in :ref:`tutorials` and :ref:`api_docs`.

* If you have some other data that you want to classify, train a network from scratch using :func:`gamornet_train_keras` / :func:`gamornet_train_tflearn` as shown in :ref:`tutorials` and :ref:`api_docs`.



If you are not sure about something, please look at this documentation carefully and contact us using the information available at :ref:`getting_help`.

.. important::

    GaMorNet is best utilized when you a large number of images to analyze. If you only have a handful of images (:math:`\sim 5`) that you want to look at in greater detail, your purposes in all probability will be served better by a 
    standalone light profile fitting code. 


.. _pdr_summary:

Summary of Public Data Release
-------------------------------
This section summarizes the different aspects of the data-products released with GaMorNet and how to use them. 


.. _module_camparison:

Keras v/s TFLearn
^^^^^^^^^^^^^^^^^^
Note that all the work in |Ghosh et. al. (2020)|_ was originally done using `TFLearn <http://tflearn.org>`_. We later used `Keras <https://keras.io>`_ 
to reproduce the same work. Thus, everything in the Public Data Release is available in two flavors -- Keras and TFLearn. 

.. important::
   Note that due to the inherent stochasticity involved in training a neural network, the results given by the Keras and TFLearn models are very close, but
   not exact replicas of one another. If you want to re-create the results in |Ghosh et. al. (2020)|_, you should use the TFLearn flavored data products. 
   In all other cases, we recommend using the Keras flavored data products as it will be better supported in the future. Look below to understand how the two flavors are different.

.. warning::
   Note that for the Keras models, the accuracies achieved are slightly different than what was achieved with TFLearn in |Ghosh et. al. (2020)|_. Additionally,
   the recommended probability thresholds are also different. Please read the information below before using the Keras models.


**Accuracies**

The accuracies achieved with the both the Keras & TFLearn models for the sample of |Ghosh et. al. (2020)|_ are shown below. These tables are similar in information 
content to Tables 5 and 7 in |Ghosh et. al. (2020)|_, which were obtained using TFLearn. 


+------------------------+------------+----------+
| **Keras on SDSS**      | Predicted  |Predicted |
|                        | Disks      |Bulges    | 
+========================+============+==========+
| **Actual Disks**       | 99.72%     | 3.37%    | 
+------------------------+------------+----------+
| **Actual Bulges**      | 0.15%      | 95.25%   |
+------------------------+------------+----------+


+------------------------+------------+----------+
| **Keras on CANDELS**   | Predicted  |Predicted |
|                        | Disks      |Bulges    | 
+========================+============+==========+
| **Actual Disks**       | 94.45%     | 21.74%   | 
+------------------------+------------+----------+
| **Actual Bulges**      | 5.37%      | 77.88%   |
+------------------------+------------+----------+


+------------------------+------------+----------+
| **TFLearn on SDSS**    | Predicted  |Predicted |
|                        | Disks      |Bulges    | 
+========================+============+==========+
| **Actual Disks**       | 99.72%     | 4.13%    | 
+------------------------+------------+----------+
| **Actual Bulges**      | 0.19%      | 94.83%   |
+------------------------+------------+----------+

+------------------------+------------+----------+
| **TFLearn on CANDELS** | Predicted  |Predicted |
|                        | Disks      |Bulges    | 
+========================+============+==========+
| **Actual Disks**       | 91.83%     | 20.86%   | 
+------------------------+------------+----------+
| **Actual Bulges**      | 7.90%      | 78.62%   |
+------------------------+------------+----------+


.. important::
    For an additional consistency-check, we counted how many of the galaxies switched classifications between disk-dominated and bulge-dominated, when predictions were performed separately using the Keras and TFLearn models. For both the SDSS and CANDELS samples, this number is :math:`\leq 0.04\%`


**Indeterminate Fraction**

The table below shows the number of galaxies in the |Ghosh et. al. (2020)|_ sample that are classified by the various models of GaMorNet to be indeterminate. This includes galaxies
which have intermediate bulge-to-total light ratios (:math:`0.45 \leq L_B/L_T \leq 0.55`) and those for which the network is not confident enough to make a prediction. For more
information, please refer to Section 4 of the paper. 


+------------------------+------------+----------+------------+----------+
|                        | Keras      |Keras     | TFLearn    |TFLearn   |
|                        | SDSS       |CANDELS   | SDSS       |CANDELS   |
+========================+============+==========+============+==========+
| Indeterminate Galaxies | 31%        | 46%      | 33%        | 39%      |
+------------------------+------------+----------+------------+----------+



**Thresholds Used**

To turn GaMorNet's output probability values into class predictions, we use probability thresholds. The probability thresholds that were used to generate the prediction tables as well as the tables above are shown below. 


*Keras on SDSS*

#. Disk-dominated if disk-probability :math:`\geq 70\%`
#. Bulge-dominated if bulge-probability :math:`\geq 70\%`
#. Indeterminate otherwise

*Keras on CANDELS*

#. Disk-dominated if disk-probability > bulge and indeterminate probability
#. Bulge-dominated if bulge-probability :math:`\geq 60\%`
#. Indeterminate otherwise

*TFLearn on SDSS*

#. Disk-dominated if disk-probability :math:`\geq 80\%`
#. Bulge-dominated if bulge-probability :math:`\geq 80\%`
#. Indeterminate otherwise


*TFLearn on CANDELS*

#. Disk-dominated if disk-probability > bulge and indeterminate probability and 36%
#. Bulge-dominated if bulge-probability :math:`\geq 55\%`
#. Indeterminate otherwise


.. important::

   The choice of the confidence/probability threshold is arbitrary and should be chosen appropriately for the particular task at hand. Towards this end, Figures 8 and 9 
   of |Ghosh et. al. (2020)|_ can be used to asses the trade-off between accuracy and completeness for both samples.

   For more information about the impact of probability thresholds on the results, please refer to Section 4.1 of the paper


.. _ftp_server:

FTP Server
^^^^^^^^^^^

All components of the public data release are hosted on the Yale Astronomy FTP server ``ftp.astro.yale.edu``. There are multiple ways you can access the FTP server
and we summarize some of the methods below.

**Using Linux Command Line** ::

    ftp ftp.astro.yale.edu
    cd pub/aghosh/<appropriate_subdirectory>

If prompted for a username, try ``anonymous`` and keep the password field blank.

**Using a Browser**

Navigate to ``ftp://ftp.astro.yale.edu/pub/aghosh/<appropriate_subdirectory>``


**Using Finder on OSX**

Open Finder, and then choose Go :math:`\Rightarrow` Connect to Server (or command + K) and enter ``ftp://ftp.astro.yale.edu/pub/aghosh/``. Choose to connect as 
``Guest`` when prompted. 

Thereafter, navigate to the appropriate subdirectory. 


.. _pred_tables:

Prediction Tables
^^^^^^^^^^^^^^^^^^
The predicted probabilities (of being disk-dominated, bulge-dominated, or indeterminate) and the final classifications for all of the galaxies 
in the SDSS and CANDELS test sets of |Ghosh et. al. (2020)|_ are made available as ``.txt``  files. 
These tables are the full versions of Tables 4 and 6 in the paper. The appropriate sub-directories of the :ref:`ftp_server` are mentioned below:-

*TFLearn*

* SDSS dataset predictions :math:`\Rightarrow` `/gamornet/pred_tables/pred_table_sdss.txt`

* CANDELS dataset predictions :math:`\Rightarrow` `/gamornet/pred_tables/pred_table_candels.txt`

*Keras*

* SDSS dataset predictions :math:`\Rightarrow` `/gamornet_keras/pred_tables/pred_table_sdss.txt`

* CANDELS dataset predictions :math:`\Rightarrow` `/gamornet_keras/pred_tables/pred_table_candels.txt`


.. _trained_models:

Trained Models
^^^^^^^^^^^^^^^
Note that the functions :func:`gamornet_predict_keras`, :func:`gamornet_predict_tflearn` automatically download and use the trained models when the correct
parameters are passed to them. However, in case you want to just download the model files for yourself, navigate to the appropriate sub-directories on the
:ref:`ftp_server` as mentioned below. For more information about these models, please refer to |Ghosh et. al. (2020)|_ and see :ref:`usage_advice`. 

*TFLearn*

* SDSS model trained only on simulations :math:`\Rightarrow` `/gamornet/trained_models/SDSS/sim_trained/`

* SDSS model trained on simulations and real data :math:`\Rightarrow` `/gamornet/trained_models/SDSS/tl/`

* CANDELS model trained only on simulations :math:`\Rightarrow` `/gamornet/trained_models/CANDELS/sim_trained/`

* CANDELS model trained on simulations and real data :math:`\Rightarrow` `/gamornet/trained_models/CANDELS/tl/`

*Keras*

* SDSS model trained only on simulations :math:`\Rightarrow` `/gamornet_keras/trained_models/SDSS/sim_trained/`

* SDSS model trained on simulations and real data :math:`\Rightarrow` `/gamornet_keras/trained_models/SDSS/tl/`

* CANDELS model trained only on simulations :math:`\Rightarrow` `/gamornet_keras/trained_models/CANDELS/sim_trained/`

* CANDELS model trained on simulations and real data :math:`\Rightarrow` `/gamornet_keras/trained_models/CANDELS/tl/`




.. |Ghosh et. al. (2020)| replace:: Ghosh et. al. (2020)
.. _Ghosh et. al. (2020): https://iopscience.iop.org/article/10.3847/1538-4357/ab8a47/pdf


