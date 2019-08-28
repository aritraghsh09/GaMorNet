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
 
start_timestamp = time.time() #to keep track of the time it takes to load the data

#Paths & GLOBAL VARIABLES
dataPath = # Point this to the relevant folder with the cutouts
modelLoadPath = '/gpfs/loomis/project/fas/urry/ag2422/galnet_runs/run_sdssg_2/check-1125600'  #Path which has the model to be loaded for transfer learning. 
OutFilePath = '/gpfs/loomis/project/fas/urry/ag2422/galnet_runs/run_sdssg_3/preds_2_all.txt'

#We run the data in batches through the netwowk so that it doesn't need to deal with 50000 or 100000 images at one go.  
BATCH_SIZE = 200 
NUM_THREADS = 10

#Preparing the Data
gal_para = plt.genfromtxt("/home/fas/urry/ag2422/project/data/sdss_btr_files/cat_sdss_btr_g.txt",dtype=[('ObjID',np.int64), ('bt_g', np.float32),('bt_g_err', np.float32),('z', np.float32),('file_name', object),('file_path', object)],skip_header=1)

def array_image(i):
	return np.reshape(fits.getdata(dataPath + gal_para["file_name"][i],memmap=False),newshape=(167,167,1))
	#This will fail when working with large files. In that case, remove this line & follow alternate approach below.
	#For large files, it might be the case that multiprocessing is not possbile due to issues below of mem leaks.

        #hdul = fits.open(dataPath+"output_img_"+str(i)+".fits")
        #hdul_data = hdul[0].data
        #X_test.append(np.reshape(hdul_data,newshape=(200,200,1)))
        #del hdul[0].data #To prevent memory leak from memmap.
        #del hdul_data #To prevent memory leak from memmap. 
        #gc.collect() #Turn this on if previous step is still leading to a memory leak


#X is the array of data(images)
pl = Pool(NUM_THREADS)
X = pl.map(array_image,range(0,len(gal_para['bt_g'])))

print("Finished Loading Data. Real Time needed to load data:- %s seconds\nProceeding to Train network......." % (time.time() - start_timestamp) )

# building AlexNet
network = input_data(shape=[None, 167, 167, 1]) #since we do training in batches, the first None serves as to empower us to choose any batch size that we may want
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

#Loading the Model which will be used for evaluation
model.load(modelLoadPath)

preds = [] #array to store results 
total_elements = len(gal_para["bt_g"])
num_batches = int(total_elements/BATCH_SIZE)

for i in range(0,num_batches):

	ll = i*BATCH_SIZE
	ul = (i+1)*BATCH_SIZE
	preds.extend(model.predict(X[ll:ul]))
	
	if (i%100) == 0:
		print(i)  #print occasional update on batch number 

if ul != len(X):
	preds.extend(model.predict(X[ul:len(X)])) #for the last partial batch

preds = np.array(preds) #converting to a numpy array for easier handling.
preds_file = open(OutFilePath,"w") #change this to .npy to store as binary

stacked_para = np.zeros(len(gal_para['bt_g']),dtype=[('ObjID',np.int64), ('bt_g', np.float32),('bt_g_err', np.float32),('z', np.float32),('file_name', object),('disk_prob',np.float32),('unclass_prob',np.float32),('ellips_prob',np.float32)])

stacked_para['ObjID'] = gal_para['ObjID']
stacked_para['bt_g'] = gal_para['bt_g']
stacked_para['bt_g_err'] = gal_para['bt_g_err']
stacked_para['z'] = gal_para['z']
stacked_para['file_name'] = gal_para['file_name']
stacked_para['disk_prob'] = preds[:,0]
stacked_para['unclass_prob'] = preds[:,1]
stacked_para['ellips_prob'] = preds[:,2]

np.savetxt(preds_file,stacked_para,delimiter=" ",header="ObjID bt_g bt_g_err z file_name disk_prob unclass_prob ellips_prob",fmt=['%d','%.2f','%.2f','%.4f','%s','%.4f','%.4f','%.4f'])

