# GaMorNet (Galaxy Morphology Network)

GaMorNet is a Convolutional Neural Network based on AlexNet to classify galaxies morphologically. GaMorNet does not need a large amount of training data (as it is trained on simulations and then transfer-learned on a small portion of real data) and can be applied on multiple datasets. Till now, GaMorNet has been tested on ~100,000 SDSS g-band galaxies and ~20,000 CANDELS H-band galaxies and has a misclassification rate of <5%. 

The training, testing and relevant statistics of GaMorNet is outlined in the paper mentioned in the next section. Please refer to the relevant sections of the paper for details about the training and transfer learning methods. 

## Usage Info/Citation/Reference
This repository was used in the work pertaining to the following research article:-
"Galaxy Morphology Network (GaMorNet):  A Convolutional Neural Network used to study morphology and quenching in ∼100,000 SDSS and ∼20,000 CANDELS galaxies", Ghosh et. al.

If you use this code for any published work, please cite the above paper and include a link to this GitHub repository.

This code is being made available under a GNU General Public License v3.0. Please see the file called LICENSE in the repository for more details.

---

## What do you need to run GaMorNet?
GaMorNet was coded in Python using [TFLearn](http://tflearn.org/) which is a high-level API for [Tensorflow](https://www.tensorflow.org/). GaMorNet has been tested to work with the following versions of different libraries mentioned below.

| Python  |  Numpy | Tensorflow-gpu  |  TFLearn  | CUDA  | cuDNN | 
|---|---|---|---|---| --- |
| 2.7.14 | 1.16.14 | 1.12.0 | 0.3.2 | 9.0.176 | 7.1.4 |
| 3.6.9  | 1.17.0  | 1.12.0  | 0.3.2  | 9.0.176 | 7.1.4 |
| 3.6.9 | 1.17.0 | 1.13.1 | 0.3.2 | 10.0.130 | 7.6.0 |

**The first configuration is what GaMorNet was originally developed in and thus is the most stable; however substantial testing has also been performed with the other two configurations**. It is known that the last configuration mentioned leads to some Tensorflow depreciation warnings.

Steps to install GaMorNet dependencies:-

* It is highly recommended to initiate a Python virtual environment (eg. using [Anaconda](https://www.anaconda.com/distribution/)) with the above-mentioned versions of Python, Numpy, TF-gpu, and TFLearn. Note that CUDA and cuDNN are necessary if you want to use GPU acceleration. More information on using Tensorflow GPU acceleration is available [here](https://www.tensorflow.org/install/gpu). Some other Python libraries that GaMorNet depends on are matplotlib, astropy, math, time and multiprocessing.

   * To initiate a new conda environment with a specific version of Python, you can do `conda create -n yourenvname python=x.x` where x.x is the Python version number and yourenvname is the name of the environment
   * Then, activate the environment using `conda activate yourenvname` or ` source activate yourenvname`
   * To install the relevant numpy version, do `conda install numpy=x.x.x` where x.x.x is the appropriate version number
   * For installing the other libraries, the following commands should suffice:-
   ```
   conda install matplotlib
   pip install astropy
   pip install multiprocessing
   ```

   * Now, install Tensorflow in the same environment. [Instructions for Installing Tensorflow](https://www.tensorflow.org/install) [**Please don't install Tensorflow in a separate environment**] 

   * Now, install TFLearn in the same environment. [Instructions for Installing TFLearn](http://tflearn.org/installation/) (Recommended way is to just do `pip install tflearn`) [**Please don't install TFLearn in a separate environment**]

   * To make sure that you have installed both TFLearn and Tensorflow correctly, run the following piece of code in an interactive Python session to verify the installation. 

   ```
   python
   import tensorflow as tf
   hello = tf.constant('Hello, TensorFlow!')
   sess = tf.Session()
   print(sess.run(hello))
   a = tf.constant(10)
   b = tf.constant(32)
   print(sess.run(a + b))

   import tflearn as tfl
   ```
   * Finally to de-activate your environment, type `conda deactivate` or `source deactivate`

* If all the above commands work, then you are all set and the gamornet scripts should run without any errors. **If there are  warnings or errors, please check to make sure that you have the recommended versions of critical libraries according to the table above**

---

## The Network

### Design -- [gamornet.py](gamornet.py)
The file [gamornet.py](gamornet.py) contains the code that we use to create the GaMorNet model in TFLearn. *Note that this file will not run without further modification -- this is only meant to show what exact TFLearn functions we used to code GaMorNet*

### Using our Trained Models --- [gamornet_predict.py](/gamornet_predict/gamornet_predict.py)
In our paper, we outline how we train GaMorNet. These trained models can be accessed via http://www.astro.yale.edu/aghosh/gamornet.html or http://gamornet.ghosharitra.com . In order to use these trained models, we provide some example code in the file [gamornet_predict.py](/gamornet_predict/gamornet_predict.py)

For this demonstration, we will be using our final trained SDSS model to predict the classification of two randomly chosen SDSS g-band images stored in the directory [sdss_cutouts](/gamornet_predict/sdss_cutouts/). Positional information on these galaxies is available in the [info.txt](/gamornet_predict/sdss_cutouts/info.txt) file.

The following steps demonstrate what you need to do, to run the script successfully. 

* Clone this GitHub Repository using the appropriate  https or ssh link
    * ```git clone https://github.com/aritraghsh09/GaMorNet.git``` 
    * OR
    * ``` git clone git@github.com:aritraghsh09/GaMorNet.git```
    
* Make sure that you have the following Python Libraries already installed besides Tensorflow and TFLearn :- numpy, pylab, astropy, math, time, multiprocessing

* Download the 3 model files using this ftp link ftp://ftp.astro.yale.edu/pub/aghosh/gamornet/trained_models/SDSS/tl/ and **store these in the [gamornet_predict](/gamornet_predict/) directory**
   * You can copy & paste the above ftp link into your browser and them manually download and place the relevant files in the correct directory.
   * OR use the following commands using a terminal
   * `cd GaMorNet/gamornet_predict/`
   * `ftp ftp.astro.yale.edu`
   * If you are prompted for a username, enter ```anonymous``` and keep the password field blank
   * After logging in, navigate to the appropriate directory using ```cd pub/aghosh/gamornet/trained_models/SDSS/tl/```
   * `get check-1546293.meta`
   * `get check-1546293.index`
   * `get check-1546293.data-00000-of-00001`
   * `quit`

* Activate the anaconda environment (where you installed the GaMorNet dependencies) using `conda activate yourenvname` or ` source activate yourenvname`

* Run the [gamornet_predict](/gamornet_predict/gamornet_predict.py) script using ```python gamornet_predict.py```

* The predicted probabilities for the test images should be written to an output file named predictions.txt


To run predictions on other images than the ones supplied, keep the following information in mind:-

* GaMorNet-SDSS & GaMorNet-CANDELS were trained for square images of 167 pixels and 83 pixels respectively. If you are using our trained models, you need to make sure the cutouts you are using are of the same size. The code will still run if you use cutouts of a different size (as the input data is reshaped to the appropriate size) but will give you erroneous results. 

* GaMorNet-SDSS & GaMorNet-CANDELS were trained for galaxies at z\~0 and z\~1 respectively. If you are using our trained models, you need to make sure the cutouts you are using are at similar redshifts. To perform predictions on galaxies at a substantially different z, you will need to retrain GaMorNet accordingly. 

* GaMorNet-SDSS & GaMorNet-CANDELS were trained with g-band and H-band images respectively. If you are using our trained models, you need to make sure the galaxy images you are using are in nearby bands. To perform predictions on galaxies at a substantially blueshifted or redshifted band, you might need to retrain GaMorNet accordingly. 

* To perform prediction on CANDELS images, you need to alter the following parameters in the code
  * In the ```array_image``` function in [gamornet_predict.py](/gamornet_predict/gamornet_predict.py) alter the ```newshape``` argument to ```newshape=(83,83,1)```
  * The input layer of the network needs to be changed to ```network = input_data(shape=[None, 83, 83, 1])```
  * You need to download the appropriate CANDELS trained models instead of the SDSS models mentioned above  
  
### (Re)Training GaMorNet -- [gamornet_train.py](/gamornet_train/gamornet_train.py)
In our paper, we outline how we first train GaMorNet on simulated images and then transfer-learn (i.e. retrain the network) on real images. 

The script, [gamornet_train.py](/gamornet_train/gamornet_train.py) is meant to demonstrate how GaMorNet can be trained. For demonstrating this, we will be using 5 simulated SDSS images from our sample. These images are stored in the folder [simulated_images](/gamornet_train/simulated_images/). The parameters used to simulate these galaxies is stored in the file [sim_para.txt](/gamornet_train/simulated_images/sim_para.txt). Using this file, we deduce the correct classification for each galaxy by using the integrated magnitudes of the disk and bulge for each galaxy. 

*Note that the script is set up to train for 5 epochs; use 3 images for training and 2 for validation. This is for demonstration purposes and will not lead to any useful results. You need to use much larger number of simulated images (~100,000) and train for many more epochs.(~500)*

The following steps demonstrate what you need to do, to run the script successfully. 

* Clone this GitHub Repository using the appropriate  https or ssh link
    * ```git clone https://github.com/aritraghsh09/GaMorNet.git``` 
    * OR
    * ``` git clone git@github.com:aritraghsh09/GaMorNet.git```
  
* Activate the anaconda environment (where you installed the GaMorNet dependencies) using `conda activate yourenvname` or ` source activate yourenvname`

* Make sure that you have the following Python Libraries already installed besides Tensorflow and TFLearn :- numpy, pylab, astropy, math, time, multiprocessing

* Run the script [gamornet_train.py](/gamornet_train/gamornet_train.py) using `python gamornet_train.py`. 

* The model file will be stored as 3 files named checkpoint-5.meta, checkpoint-5.index, checkpoint-5.data-00000-of-00001 in the /gamornet_train/ directory. The number 5 indicates that the model was trained for 5 epochs.


#### To retrain starting from one of our models
In this demo., we will be retraining GaMorNet using the above mentioned simulated images starting from our final SDSS model. *Retraining our final model on simulated images doesn't make any scientific sense and is just for demonstration purposes.* 

* Download the 3 model files using this ftp link ftp://ftp.astro.yale.edu/pub/aghosh/gamornet/trained_models/SDSS/tl/ and **store these in the [gamornet_train](/gamornet_train/) directory**
   * You can copy & paste the above ftp link into your browser and then manually download and place the relevant files in the correct directory.
   * OR use the following commands using a terminal
   * `cd GaMorNet/gamornet_train/`
   * `ftp ftp.astro.yale.edu`
   * If you are prompted for a username, enter ```anonymous``` and keep the password field blank
   * After logging in, navigate to the appropirate directory using ```cd pub/aghosh/gamornet/trained_models/SDSS/tl/```
   * `get check-1546293.meta`
   * `get check-1546293.index`
   * `get check-1546293.data-00000-of-00001`
   * `quit`
   
* Activate the anaconda environment (where you installed the GaMorNet dependencies) using `conda activate yourenvname` or ` source activate yourenvname`

* Uncomment Line #138 in the GaMorNet code ``` #model.load(modelLoadPath+"check-1546293")```

* If you want to freeze/lock some layers during training, add the `trainable=False` argument to the layer you want to lock. For eg. to lock Layer 8, uncomment the following line `#network = conv_2d(network, 384, 3, activation='relu',trainable=False)  ` and **comment out the original layer 8 (Line #105)**

* If you don't want load some layers from the model and want to initialize these from scratch, add the `restore=False` argument to the layer you don't want to load. For eg. to not load Layer 13, uncomment the following line ` #network = fully_connected(network, 4096, activation='tanh',restore=False)` and **comment out the original layer 13 (Line #115)**

* Run the script [gamornet_train.py](/gamornet_train/gamornet_train.py) using `python gamornet_train.py`. 

* The model file will be stored as 3 files named checkpoint-1546298.meta, checkpoint-1546298.index, checkpoint-1546298.data-00000-of-00001 in the /gamornet_train/ directory. The number 1546298 indicates that the model was trained for 5 epochs since 1546293

---
## Where are all the Model Files? 
The Base Directory for all the models is [ftp://ftp.astro.yale.edu/pub/aghosh/gamornet/trained_models/](ftp://ftp.astro.yale.edu/pub/aghosh/gamornet/trained_models/)

After that the different models are arranged as follows:-
* GaMorNet-S model trained only on simulations &rightarrow; /SDSS/sim_trained/
* GaMorNet-S model trained on simulations and then transfer learned on real data &rightarrow; /SDSS/tl/
* GaMorNet-C model trained only on simulations &rightarrow; /CANDELS/sim_trained/
* GaMorNet-C model trained on simulations and then transfer learned on real data &rightarrow; /CANDELS/tl/


**For other products of the public data release, please refer to the Appendix of the paper or head to [this link](http://www.astro.yale.edu/aghosh/gamornet.html) or [this link](http://gamornet.ghosharitra.com).**

--- 
## Questions?
* If you have an upgrade idea or some way to make the code more efficient, please go ahead and submit a pull request.
* If you are sure you followed the installation instructions correctly and there is something wrong with the instructions, please open up an issue on GitHub or submit a pull request with relevant changes to the read-me.
* If you are not sure about how to use something, please send me an email at aritraghsh09+gamornet at gmail.com 
