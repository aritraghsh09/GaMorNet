# GaMorNet

GaMorNet is a Convolutional Neural Network based on AlexNet to classify galaxies morphologically. GaMorNet does not need a large amount of training data (as it is trained on simulations and then transfer-learned on a small portion of real data) and can be applied on multiple datasets. Till now, GaMorNet has been tested on ~100,000 SDSS g-band galaxies and ~20,000 CANDELS H-band galaxies and has a misclassification rate of <5%. 

The training, testing and relevant statistics of GaMorNet is outlined in the paper mentioned in the next section. Please refer to the relevant sections of the paper for details about the training and transfer learning methods. 

## Usage Info/Citation/Reference
This repository was used in the work pertaining to the following research article:-
"Galaxy Morphology Network (GaMorNet):  A Convolutional Neural Network used to study morphology andquenching in ∼100,000 SDSS and ∼20,000 CANDELS galaxies" , Ghosh et. al.

If you use this code for any published work, please cite the above paper and include a link to this GitHub repository.

This code is being made available under a GNU General Public License v3.0. Please see the file called LICENSE in the repository for more details.

---

## What do you need to run GaMorNet?
GaMorNet was trained and tested using [TFLearn](http://tflearn.org/) which is a high-level API for [Tensorflow](https://www.tensorflow.org/). GaMorNet has been tested to work with the following versions of different libraries mentioned below.

| Python  |  Numpy | TF-gpu  |  TFLearn  | CUDA  | cuDNN | 
|---|---|---|---|---| --- |
| 2.7.14 | 1.16.14 | 1.12.0 | 0.3.2 | 9.0.176 | 7.1.4 |
| 3.6.9  | 1.17.0  | 1.12.0  | 0.3.2  | 9.0.176 | 7.1.4 |
| 3.6.9 | 1.17.0 | 1.13.1 | 0.3.2 | 10.0.130 | 7.6.0 |

The last confifuration mentioned will lead to depreciation warnings and might lead to errors depedning on other Python Libraries installed on your machines. Thus, the first two configurations are the recommended configurations. 

It is highly recommended to initiate a Python virtual environment (eg. using Anaconda) with the above mentioned versions of Python, Numpy, TF-gpu and TFLearn. Note that CUDA and cuDNN are necessary if you want to use GPU acceleration. More information of using Tensorflow GPU acceleration is available [here](https://www.tensorflow.org/install/gpu)

* [Instructions for Installing Tensorflow](https://www.tensorflow.org/install)
* [Instructions for Installing TFLearn](http://tflearn.org/installation/) Recommended way is to just do `pip install tflearn`

Once you are sure that you have installed both TFLearn and Tensorflow correctly, run the following piece of code in an interactive session to verify the installation. 

```
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))

import tflearn as tfl
```

If all the above commands work, then you are all set. **If there warnings or errors, please check to make sure that you have the recommended versions of critical libraries according to the table above**

---

## The Network

### Design
The file [gamronet.py](gamornet.py) contains the code that we use to create the GaMorNet model in TFLearn. *Note that this file will not run without further modification -- this is only meant to show what exact TFLearn functions we used to code GaMorNet*

### Using our Trained Models
In our paper, we outline how we train GaMorNet. These trained models can be accessed via http://www.astro.yale.edu/aghosh/gamornet.html or http://gamornet.ghosharitra.com . In order to use these trained models, we provide some example code in the file [gamornet_use_trained.py](gamornet_used_trained.py)

This 

