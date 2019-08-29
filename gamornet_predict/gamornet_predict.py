########################################
#  gamornet_predict.py
#
#  This script is meant to use the trained models to get predictions on real galaxies.
#  This script stores the prediction probabilites for each bin/classification in a file. 
# --------------------------------------
# THIS SCRIPT DOES NOT RUN BY ITSELF. TO SEE HOW TO RUN THIS SCRIPT, REFER TO THE README OF THIS REPOSITORY. 
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
dataPath = './sdss_cutouts/' #Folder with the images to be tested 
modelLoadPath = './check-1546293'#Path to model file you are using to make the predictions
OutFilePath = './predictions.txt' #Output file containing the predictions by GaMorNet for each input image

#To optimize RAM usage while performing prediction on a large number of images, We run the data in batches through the netwowk and also load the data in parallel. The following two parameters deal with this./
BATCH_SIZE = 1  #Number of images to be fed in a single batch 
NUM_THREADS = 1  #Number of threads to be initiated while loading the data

#Details of images to be fed in
gal_para = plt.genfromtxt(dataPath + "info.txt", usecols=(0,1), names=True, dtype=None, encoding=None)

#The array_image function is to help return an array of images on which the network will perform prediction
def array_image(i):
	return np.reshape(fits.getdata(dataPath + gal_para["file_name"][i],memmap=False),newshape=(167,167,1)) 
	##!! NOTE HERE THAT THIS FUNCTION RESHAPES THE DATA TO WHAT GaMorNet-SDSS WAS TRAINED FOR. USING AN IMAGE WITH A VERY DIFFERENT SHAPE AND TRYING TO RESHAPE IT WILL LEAD TO ERRORNIOUS RESULTS. !!##
	

	#The function above will fail when working with large files. In that case, remove this line & follow alternate approach below.
	#For large files, it might also be the case (depending on your system) that multiprocessing is not possbile due to issues below of mem leaks.
        #hdul = fits.open(dataPath+"output_img_"+str(i)+".fits")
        #hdul_data = hdul[0].data
        #X_test.append(np.reshape(hdul_data,newshape=(200,200,1)))
        #del hdul[0].data #To prevent memory leak from memmap.
        #del hdul_data #To prevent memory leak from memmap. 
        #gc.collect() #Turn this on if previous step is still leading to a memory leak


#Now, we create an array, X is the array of data(images) using miltiple threads to speed up the process of data-loading
pl = Pool(NUM_THREADS)
X = pl.map(array_image,range(0,len(gal_para['file_name'])))

print("Finished Loading Data. Real Time needed to load data:- %s seconds\nProceeding to Train network......." % (time.time() - start_timestamp) )




##-----------------BUILDING AND LOADING THE MODEL -----------------##


# building GaMorNet -- Refer to gamornet.py for more details about individual layers. 
network = input_data(shape=[None, 167, 167, 1]) #since we feed the data in batches, the first None serves as to empower us to choose any batch size that we may want
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 3, activation='softmax')

network = regression(network, optimizer='momentum',loss='categorical_crossentropy',learning_rate=0.0001)
model = tflearn.DNN(network)

#Loading the Model which will be used for evaluation/prediction
model.load(modelLoadPath)




## ------------- PERFORMING PREDICTION ON LOADED IMAGES ----------------##


preds = [] #array to store results/predictions
total_elements = len(gal_para["file_name"])
num_batches = int(total_elements/BATCH_SIZE)

#if statement to make sure the number of batches is atleast 1. 
if num_batches < 1:
	print("ERROR: Number of Batches Must at Least be 1. Exiting......")
	exit()

#Now we use a for loop to call model.predict on each batch of images on which prediction is to be performed. 
for i in range(0,num_batches):

	ll = i*BATCH_SIZE #lower index 
	ul = (i+1)*BATCH_SIZE #upper index
	preds.extend(model.predict(X[ll:ul]))
	
	if (i%100) == 0:
		print(i)  #print occasional update on batch number 

#The following if statement checks for the last partial batch if any and runs model.predict on those images. 
if ul != len(X):
	preds.extend(model.predict(X[ul:len(X)])) 




## ------------- WRITING THE RESULTS TO AN OUTPUT FILE ------------------##


preds = np.array(preds) #converting to a numpy array for easier handling.
preds_file = open(OutFilePath,"w") #Chande the ending of the output file to .npy to store as binary

#Creating a stacked array for writing to the output file
stacked_para = np.zeros(len(gal_para['file_name']),dtype=[('ObjID',np.int64),('file_name', object),('disk_prob',np.float32),('unclass_prob',np.float32),('ellips_prob',np.float32)])

stacked_para['ObjID'] = gal_para['ObjID']
stacked_para['file_name'] = gal_para['file_name']
stacked_para['disk_prob'] = preds[:,0]
stacked_para['unclass_prob'] = preds[:,1]
stacked_para['ellips_prob'] = preds[:,2]

#Writing to output file
np.savetxt(preds_file,stacked_para,delimiter=" ",header="ObjID file_name disk_prob unclass_prob ellips_prob",fmt=['%d','%s','%.4f','%.4f','%.4f'])

