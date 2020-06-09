.. _usage_guide:

PDR Usage Guide
===============

If you are looking for information about the various ways you can use GaMorNet (running on a CPU v/s GPU v/s the cloud) or installation instructions, please have a look at 
:ref:`getting_started`. This section summarizes different aspects of the public data release and provides some advice on the applicability of GaMorNet for various tasks. 



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
   not completely exact replicas of one another. If you want to re-create the results in |Ghosh et. al. (2020)|_, you should use the TFLearn flavored data products. 
   In all other cases, we recommend using the Keras flavored data products as it will be better supported in the future. Look below for how the two flavors are different.

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
| Actual Disks           | 99.72%     | 3.37%    | 
+------------------------+------------+----------+
| Actual Bulges          | 0.15%      | 95.25%   |
+------------------------+------------+----------+


+------------------------+------------+----------+
| **Keras on CANDELS**   | Predicted  |Predicted |
|                        | Disks      |Bulges    | 
+========================+============+==========+
| Actual Disks           | 94.45%     | 21.74%   | 
+------------------------+------------+----------+
| Actual Bulges          | 5.37%      | 77.88%   |
+------------------------+------------+----------+


+------------------------+------------+----------+
| **TFLearn on SDSS**    | Predicted  |Predicted |
|                        | Disks      |Bulges    | 
+========================+============+==========+
| Actual Disks           | 99.72%     | 4.13%    | 
+------------------------+------------+----------+
| Actual Bulges          | 0.19%      | 94.83%   |
+------------------------+------------+----------+

+------------------------+------------+----------+
| **TFLearn on CANDELS** | Predicted  |Predicted |
|                        | Disks      |Bulges    | 
+========================+============+==========+
| Actual Disks           | 91.83%     | 20.86%   | 
+------------------------+------------+----------+
| Actual Bulges          | 7.90%      | 78.62%   |
+------------------------+------------+----------+


.. important::
    We additionally checked, how many of the galaxies switched classifications between disk-dominated and bulge-dominated, when predictions were performed separately using 
    the Keras and TFLearn models. For both the SDSS and CANDELS samples, this number is :math:`\leq 0.04\%`


**Indeterminate Fraction**

The table below shows the number of galaxies in the |Ghosh et. al. (2020)|_ sample that are classified by the various models of GaMorNet to be indeterminate. This includes galaxies
which have intermediate bulge-to-total light ratios (:math:`0.45 \leq L_B/L_T \leq 0.55`) and those for which the network is not confident enough to make a prediction. For more
information please refer to Section 4 of the paper. 


+------------------------+------------+----------+------------+----------+
|                        | Keras      |Keras     | TFLearn    |TFLearn   |
|                        | SDSS       |CANDELS   | SDSS       |CANDELS   |
+========================+============+==========+============+==========+
| Indeterminate Galaxies | 31%        | 46%      | 33%        | 39%      |
+------------------------+------------+----------+------------+----------+



**Thresholds Used**

The probability thresholds that were used to generate the prediction tables as well as the tables above are shown below. 


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

   The choice of the confidence threshold is arbitrary and should be chosen appropriately for the particular task at hand. Toward this end, Figures 8 and 9 
   of |Ghosh et. al. (2020)|_ can be used to asses the trade-off between accuracy and completeness for both samples.

   For more information about the impact of probability thresholds on the results, please refer to Section 4.1 of the paper



.. _pred_tables:

Prediction Tables
^^^^^^^^^^^^^^^^^^
The predicted probabilities (of being disk-dominated, bulge- dominated, or indeterminate) and the final classifications for all of the galaxies 
in the SDSS and CANDELS test sets in Ghosh et. al. 2020, as determined by GAMORNET-S and -C, are made available as .txt files. 
These tables are the full versions of Tables 4 and 6 in the paper. 


.. _trained_models:

Trained Models
^^^^^^^^^^^^^^^


.. _usage_advice:

Usage Advice
-------------
Here is a brief summary of the different ways you can use the 


.. |Ghosh et. al. (2020)| replace:: Ghosh et. al. (2020)
.. _Ghosh et. al. (2020): https://iopscience.iop.org/article/10.3847/1538-4357/ab8a47/pdf


