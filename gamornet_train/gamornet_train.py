########################################
#  gamornet_train.py
#
#  This script is meant to be used if you want re-train GaMorNet either from scratch or from a model file.
#  This script ouputs the final model after training along with printing accuracy info to stdout
# --------------------------------------
#  FOR MORE DETAILS ABOUT THIS SCRIPT, REFER TO THE README OF THIS REPOSITORY. 
########################################

from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
import pylab as plt
from astropy.io import fits
import math
import time
from multiprocessing import Pool
 



##---------------------- DATA LOADING ---------------------------##


start_timestamp = time.time() #to keep track of the time it takes to load the data

#Paths & Global variables
dataPath = './simulated_images/' #Folder with the images to be trained on 
modelLoadPath = './' #Path to model file from where you want to start re-training [NOT APPLICABLE IF YOU WANT TO START TRAINING FROM SCRATCH]
modelSavePath = './checkpoint' #Path to save the final model file

#To optimize RAM usage while training on a large number of images, we load the data in parallel.
NUM_THREADS = 1

#The following two parameters help to split the training data into a training set and a validation set.
NUM_FILES_TOTAL = 5 #Total Number of Files
NUM_FILES_TEST = 2  #Number of files you wish to use for validation. Files at the beginning of the naming sequence will be used for testing and the rest for training.


#Details of the images being fed in
gal_para = plt.genfromtxt(dataPath + "sim_para.txt", skip_header=11,names=True, dtype=None, encoding=None)
disk_bulge_mag = gal_para["Inte_Mag"] - gal_para["Inte_Mag_2"] #difference between the integrated magnitudes of disk and bulge components


#The array_image function is to help return an array of images
def array_image(i):
	return np.reshape(fits.getdata(dataPath+"output_img_"+str(i)+".fits" ,memmap=False),newshape=(167,167,1)) 
	##!! NOTE HERE THAT THIS FUNCTION RESHAPES THE DATA TO WHAT GaMorNet-SDSS WAS TRAINED FOR. USING AN IMAGE WITH A VERY DIFFERENT SHAPE AND TRYING TO RESHAPE IT WILL LEAD TO ERRORNIOUS RESULTS. !!##
	

	#The function above will fail when working with large files. In that case, remove this line & follow alternate approach below.
	#For large files, it might also be the case (depending on your system) that multiprocessing is not possbile due to issues below of mem leaks.
        #hdul = fits.open(dataPath+"output_img_"+str(i)+".fits")
        #hdul_data = hdul[0].data
        #X_test.append(np.reshape(hdul_data,newshape=(200,200,1)))
        #del hdul[0].data #To prevent memory leak from memmap.
        #del hdul_data #To prevent memory leak from memmap. 
        #gc.collect() #Turn this on if previous step is still leading to a memory leak


#The array target function returns a vector for each image corresponding to its `correct' classification inferred from the parameters of the `gal_para' file
#Since we have 3 possible classifications, each target vector has 3 entries/spaces with each entry either being 0 or 1
def array_target(i):

	target_vect = [0]*3 #Target vector to be returned

	if (disk_bulge_mag[i] < -0.22):
		target_vect[0] = 1         #disk
	elif ( -0.22 <=  disk_bulge_mag[i] <= 0.22):
		target_vect[1] = 1         #neither
	else:
		target_vect[2] = 1         #bulge

	return target_vect

#Now, we create a few arrays using miltiple threads to speed up the process of data-loading
pl = Pool(NUM_THREADS)
X = pl.map(array_image,range(NUM_FILES_TEST,NUM_FILES_TOTAL))	#Array of Training Images
Y = pl.map(array_target,range(NUM_FILES_TEST,NUM_FILES_TOTAL))	#Target Vectors corresponding to the training images

X_test = pl.map(array_image,range(0,NUM_FILES_TEST))	#Array of Validation Images
Y_test = pl.map(array_target,range(0,NUM_FILES_TEST))	#Target Vectors corresponding to the validation iamges


print("Finished Loading Data. Real Time needed to load data:- %s seconds\nProceeding to Train network......." % (time.time() - start_timestamp) )




##-----------------BUILDING AND (RE)TRAINING THE MODEL -----------------##


# building GaMorNet -- Refer to gamornet.py for more details about individual layers. 
network = input_data(shape=[None, 167, 167, 1]) #since we feed the data in batches, the first None serves as to empower us to choose any batch size that we may want
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')

#Optional
#If retraining from a previous model, you might want to `freeze' some layers and make them not trainable. In that case, the last layer will look as below
#network = conv_2d(network, 384, 3, activation='relu',trainable=False)

network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')

#Optional
#If retraining from a previous model, you might want to not load some layers from the model and instead initiate those from scratch. In that case, the last layer will look as below
#network = fully_connected(network, 4096, activation='tanh',restore=False)


network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 3, activation='softmax')

#training method and parameters.
#THE LEARNING RATE PARAMETER IS A CRITICAL PARAMETER AND ITS IDEAL VALUE DEPENDS ON THE SPECIFIC TRAINING SCENARIO. We found values around 10^-3,10^-4 to be optimal.
#Optimizer controls the optimization algorithm being used and loss controls the loss function that is being minimized. 
#The max_checkpoints argument controls the number of models you want to save. For eg. setting it to 2 saves the two most recent models. 
network = regression(network, optimizer='momentum',loss='categorical_crossentropy',learning_rate=0.0001)
model = tflearn.DNN(network, checkpoint_path=modelSavePath, max_checkpoints=1, tensorboard_verbose=0)


#Optional Step to Load a Model in case you are not training from scratch. 
#Typically, the modelLoadPath should end with something like `check-123456'. By setting it as such, all the 3 files .meta, .index and .data are read in.

#model.load(modelLoadPath+"check-1546293")

#Final Training Step
#n_epoch controls the number of epochs of training
#The shuffle Boolean Arugment indicates whether the images in each training batch are shuffled or not before being fed in
#Batch Size Controls the number of images in each batch
#The two snapshot arguments control whether a checkpoint is saved at the end of each step or at the end of each epoch
#The show metric arugment controls whether the accuracy and loss of the model is streamed to stdout during training. 
model.fit(X, Y, n_epoch=5, validation_set=(X_test,Y_test), shuffle=True, show_metric=True, batch_size=64, snapshot_step=None, snapshot_epoch=True, run_id='gamornet_demo')

