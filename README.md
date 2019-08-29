# GaMorNet

GaMorNet is a Convolutional Neural Network based on AlexNet to classify galaxies morphologically. GaMorNet does not need a large amount of training data (as it is trained on simulations and then transfer-learned on a small portion of real data) and can be applied on multiple datasets. Till now, GaMorNet has been tested on ~100,000 SDSS g-band galaxies and ~20,000 CANDELS H-band galaxies and has a misclassification rate of <5%. 

The training, testing and relevant statistics of GaMorNet is outlined in the paper mentioned in the next section. Please refer to the relevant sections of the paper for details about the training and transfer learning methods. 

## Usage Info/Citation/Reference
This repository was used in the work pertaining to the following research article:-
"Galaxy Morphology Network (GaMorNet):  A Convolutional Neural Network used to study morphology and quenching in ∼100,000 SDSS and ∼20,000 CANDELS galaxies", Ghosh et. al.

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

**The first configuration is what GaMorNet was originially developed in and thus is the most stable; however substantial testing has also been performed with the other two configurations**

The last configuration mentioned will lead to deprecation warnings and might lead to errors depending on other Python Libraries installed on your machines. Thus, the first two configurations are the recommended configurations. 

Steps to install GaMorNet dependencies:-

* It is highly recommended to initiate a Python virtual environment (eg. using Anaconda) with the above-mentioned versions of Python, Numpy, TF-gpu, and TFLearn. Note that CUDA and cuDNN are necessary if you want to use GPU acceleration. More information on using Tensorflow GPU acceleration is available [here](https://www.tensorflow.org/install/gpu)

* [Instructions for Installing Tensorflow](https://www.tensorflow.org/install)

* [Instructions for Installing TFLearn](http://tflearn.org/installation/) Recommended way is to just do `pip install tflearn`

* Now the GaMorNet python scripts should executable

To make sure that you have installed both TFLearn and Tensorflow correctly, run the following piece of code in an interactive session to verify the installation. 

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

### Design -- [gamronet.py](gamornet.py)
The file [gamronet.py](gamornet.py) contains the code that we use to create the GaMorNet model in TFLearn. *Note that this file will not run without further modification -- this is only meant to show what exact TFLearn functions we used to code GaMorNet*

### Using our Trained Models --- [gamronet_predict.py](/gamornet_predict/gamornet_predict.py)
In our paper, we outline how we train GaMorNet. These trained models can be accessed via http://www.astro.yale.edu/aghosh/gamornet.html or http://gamornet.ghosharitra.com . In order to use these trained models, we provide some example code in the file [gamornet_predict.py](/gamornet_predict/gamornet_predict.py)

For this demonstration, we will be using our final trained SDSS model to predict the classification of two randomly chosen SDSS g-band images stored in the directory [sdss_cutouts](/gamronet_predict/sdss_cutouts/). Positional information on these galaxies is available in the [info.txt](/gamronet_predict/sdss_cutouts/info.txt) file.

The following steps demonstrate what you need to do, to run the script successfully. 

* Clone this GitHub Repository using the appropriate  https or ssh link
    * ```git clone https://github.com/aritraghsh09/GaMorNet.git``` 
    * OR
    * ``` git clone git@github.com:aritraghsh09/GaMorNet.git```
    
* Make sure that you have the following Python Libraries already installed besides Tensorflow and TFLearn :- numpy,pylab,astropy, math, time,multiprocessing

* Download the 3 model files using this ftp link ftp://ftp.astro.yale.edu/pub/aghosh/gamornet/trained_models/SDSS/tl/ and store these in the [gamornet_predict](/gamornet_predict/) directory
   * You can copy & paste the above ftp link into your browser and them manually download and place the relevant files in the correct directory.
   * OR use the following commands using a terminal
   * `cd GaMorNet/gamornet_predict/`
   * `ftp ftp.astro.yale.edu`
   * If you are prompted for a username, enter ```anonymous``` and keep the password field blank
   * After logging in, navigate to the appropirate directory using ```cd pub/aghosh/gamornet/trained_models/SDSS/tl/```
   * `get check-1546293.meta`
   * `get check-1546293.index`
   * `get check-1546293.data-00000-of-00001`
   * `quit`

* Run the [gamornet_predict](/gamornet_predict/gamornet_predict.py) script using ```python gamornet_predict.py```

* The predicted probabilities for the test images should be written to an output file named predictions.txt


To run predictions on other images than the ones supplied keep the following information in mind:-

* GaMorNet-SDSS & GaMorNet-CANDELS were trained for square images of 167 pixels and 83 pixels respectively. If you are using our trained models, you need to make sure the cutouts you are using are of the same size. The code will still run if you use cutouts of a different size (as the input data is reshaped to the appropriate size) but will give you erroneous results. 

* GaMorNet-SDSS & GaMorNet-CANDELS were trained for galaxies at z\~0 and z\~1 respectively. If you are using our trained models, you need to make sure the cutouts you are using are at similar redshifts. To perform predictions on galaxies at a substantially different z, you will need to retrain GaMorNet accordingly. 

* GaMorNet-SDSS & GaMorNet-CANDELS were trained with g-band and H-band images respectively. If you are using our trained models, you need to make sure the galaxy images you are using are in nearby bands. To perform predictions on galaxies at a substantially blueshifted or redshifted band, you might need to retrain GaMorNet accordingly. 

* To perform prediction on CANDELS images, you need to alter the following parameters in the code
  * In the ```array_image``` function in [gamornet_predict.py](/gamornet_predict/gamornet_predict.py) alter the ```newshape``` argument to ```newshape=(83,83,1)```
  * The input layer of the network needs to be changed to ```network = input_data(shape=[None, 83, 83, 1])```
  * You need to download the appropriate CANDELS trained models instead of the SDSS models mentioned above
---
## Important Things to Keep in Mind


---
## Where are all the Model Files? 
The Base Directory for all the models is 

After that the different models are arranged as follows:-
* GaMorNet-S model trained only on simulations &rightarrow; /SDSS/sim\textunderscore trained/
* GaMorNet-S model trained on simulations and then transfer learned on real data &rightarrow; /SDSS/tl/
* GaMorNet-C model trained only on simulations &rightarrow; /CANDELS/sim\textunderscore trained/
* GaMorNet-C model trained on simulations and then transfer learned on real data &rightarrow; /CANDELS/tl/


**For other products of the public data release, please refer to the Appendix of the paper or head to [this link](http://www.astro.yale.edu/aghosh/gamornet.html) or [this link](http://gamornet.ghosharitra.com).**
