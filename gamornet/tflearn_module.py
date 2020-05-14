from keras.models import Sequential, load_model, clone_model
from keras.layers import Dense, Flatten, Dropout, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import VarianceScaling
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
import wget
import random
import string
import os
import numpy as np


import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import wget
import random
import string
import os
import numpy as np
import progressbar
from keras import backend as K
import time

############################################
##########HELPER FUNCTIONS##################

def check_input_shape_validity(input_shape):
	
	if input_shape == 'SDSS':
		input_shape = (167,167,1)
	elif input_shape == 'CANDELS':
		input_shape = (83,83,1)		  
	else:
		try:
			if len(input_shape) != 3:
				raise Exception("input_shape has to be a tuple (typically of 3 dimensions) or any of the available keywords")
		except:
			raise Exception("input_shape has to be a tuple (typically of 3 dimensions) or any of the available keywords")

	return input_shape


def check_imgs_validity(img_array):

	if isinstance(img_array,np.ndarray):
		if len(img_array.shape) != 4:
			raise Exception("The Image Array needs to have 4 dimensions. (num,x,y,bands)")
	elif isinstance(img_array,list):
		if len(np.array(img_array).shape) != 4:
			raise Exception("The Image Array needs to have 4 dimensions. (num,x,y,bands)")
	else:
		raise Exception("The Image Array Needs to be a 4 Dimensional Numpy Array.")


def check_labels_validity(labels):

	if isinstance(labels,np.ndarray):
		if labels.shape[1] != 3:
			raise Exception("The Labels Array needs to have 2 dimensions. (num,3)")
	elif isinstance(labels,list):
		if np.array(labels).shape[1] != 3:
			raise Exception("The Labels Array needs to have 2 dimensions. (num,3)")
	else:
		raise Exception("The Lables Array Needs to be a 2 Dimensional Numpy Array. (num,3)")


def check_bools_validity(bools):

	if (bools == 'train_bools_SDSS'):
		bools = [True]*8
	elif (bools == 'train_bools_CANDELS'):
		bools = [False,False,False,True,True,True,True,True]
	elif (bools == 'load_bools_SDSS'):
		bools = [True,True,True,True,True,False,False,False]
	elif (bools == 'load_bools_CANDELS'):
		bools = [True,True,True,True,True,True,False,False]

	try:
		for element in bools:
			if type(element) != bool:
				raise Exception("The Supplied Array of Bools doesn't look okay")
	
		if len(bools) != 8:
				raise Exception("The Supplied Array of Bools doesn't look okay")

	except:
		raise Exception("The Supplied Array of Bools doesn't look okay")

	return bools


def get_model_from_link_tflearn(base_link,file_names,model):
	
	#Skipping over random name generation because the tflearn files themselves are named in a super specific way
	#Thus there is no chance of accidental deletion like the keras model.h5 names
	
	file_links = np.core.defchararray.add(np.array([base_link]*3),file_names)
	
	for link in file_links:
		wget.download(link)

	try:
		model.load(file_names[2][:-5])
	except:
		os.remove(file_names[0])
		os.remove(file_names[1])
		os.remove(file_names[2])
		raise

	
	os.remove(file_names[0])
	os.remove(file_names[1])
	os.remove(file_names[2])
	
	return model


def gamornet_load_model_tflearn(model,model_load_path):
	
	if model_load_path == 'SDSS_sim':
		print("Fetching SDSS Sim Trained Weigths.....")
		base_link = 'ftp://ftp.astro.yale.edu/pub/aghosh/gamornet/trained_models/SDSS/sim_trained/'
		file_names = np.array(['check-1405593.data-00000-of-00001','check-1405593.index','check-1405593.meta'])
		model = get_model_from_link_tflearn(base_link,file_names,model)
		
	elif model_load_path == 'SDSS_tl':
		print("Fetching SDSS TL Weigths.....")
		base_link = 'ftp://ftp.astro.yale.edu/pub/aghosh/gamornet/trained_models/SDSS/tl/'
		file_names = np.array(['check-1546293.data-00000-of-00001','check-1546293.index','check-1546293.meta'])
		model = get_model_from_link_tflearn(base_link,file_names,model)
		
	elif model_load_path == 'CANDELS_sim':
		print("Fetching CANDELS Sim Trained Weigths.....")
		base_link = 'ftp://ftp.astro.yale.edu/pub/aghosh/gamornet/trained_models/CANDELS/sim_trained/'
		file_names = np.array(['check-562800.data-00000-of-00001','check-562800.index','check-562800.meta'])
		model = get_model_from_link_tflearn(base_link,file_names,model)
		
	elif model_load_path == 'CANDELS_tl':
		print("Fetching CANDELS TL Weigths.....")
		base_link = 'ftp://ftp.astro.yale.edu/pub/aghosh/gamornet/trained_models/CANDELS/tl/'
		file_names = np.array(['check-571275.data-00000-of-00001','check-571275.index','check-571275.meta'])
		model = get_model_from_link_tflearn(base_link,file_names,model)
		
	else:
		 model.load(model_load_path)
			
	return model

	
###########################################
###########################################



############################################
##########TFLEARN FUNCTIONS##################

def gamornet_build_model_tflearn(input_shape,trainable_bools = [True]*8,load_layers_bools = [True]*8):
	
	input_shape = check_input_shape_validity(input_shape)	

	model = input_data(shape=[None, input_shape[0], input_shape[1], input_shape[2]]) 
	model = conv_2d(model, 96, 11, strides=4, activation='relu',trainable=trainable_bools[0],restore=load_layers_bools[0])
	model = max_pool_2d(model, 3, strides=2)
	model = local_response_normalization(model)
	model = conv_2d(model, 256, 5, activation='relu',trainable=trainable_bools[1],restore=load_layers_bools[1])
	model = max_pool_2d(model, 3, strides=2)
	model = local_response_normalization(model)
	model = conv_2d(model, 384, 3, activation='relu',trainable=trainable_bools[2],restore=load_layers_bools[2])
	model = conv_2d(model, 384, 3, activation='relu',trainable=trainable_bools[3],restore=load_layers_bools[3])
	model = conv_2d(model, 256, 3, activation='relu',trainable=trainable_bools[4],restore=load_layers_bools[4])
	model = max_pool_2d(model, 3, strides=2)
	model = local_response_normalization(model)
	model = fully_connected(model, 4096, activation='tanh',trainable=trainable_bools[5],restore=load_layers_bools[5])
	model = dropout(model, 0.5)
	model = fully_connected(model, 4096, activation='tanh',trainable=trainable_bools[6],restore=load_layers_bools[6])
	model = dropout(model, 0.5)
	model = fully_connected(model, 3, activation='softmax',trainable=trainable_bools[7],restore=load_layers_bools[7])
	
	return model



def gamornet_predict_tflearn(img_array,model_load_path,input_shape,batch_size=64,individual_arrays=False,trainable_bools = [True]*8,clear_session=False):

	#TFLearn Loads graphs from memory by name, hence it's always advisable to set this to True if using in a Notebook.	
	if clear_session == True:
		K.clear_session()

	check_imgs_validity(img_array)

	trainable_bools = check_bools_validity(trainable_bools)	

	model = gamornet_build_model_tflearn(input_shape=input_shape,trainable_bools=trainable_bools)
	model = tflearn.DNN(model)	

	print("Loading GaMorNet Model.....")
	model = gamornet_load_model_tflearn(model,model_load_path)
	
	print("Performing Predictions.....")

	preds = [] #array to store results 

	total_elements = len(img_array)
	num_batches = int(total_elements/batch_size)
	
	time.sleep(0.1) #Time to Flush all Print Statements before the progressbar output comes on
	for i in progressbar.progressbar(range(0,num_batches)):
		ll = i*batch_size
		ul = (i+1)*batch_size
		preds.extend(model.predict(img_array[ll:ul]))


	if ul != len(img_array):
		preds.extend(model.predict(img_array[ul:len(img_array)])) #for the last partial batch
	

	preds = np.array(preds) #converting to a numpy array for easier handling.

	if individual_arrays == True:
		return preds[:,0],preds[:,1],preds[:,2] # 'disk_prob','unclass_prob','bulge_prob'
	else:
		return preds



def gamornet_train_tflearn(training_imgs,training_labels,validation_imgs,validation_labels,input_shape,files_save_path="./",epochs=100,checkpoint_freq=0,batch_size=64,lr=0.0001,momentum=0.9,decay=0.0,nesterov=False,loss='categorical_crossentropy',load_model=False,model_load_path="./",save_model=True,verbose=1):

	check_imgs_validity(training_imgs)
	check_imgs_validity(validation_imgs)
	check_labels_validity(training_labels)
	check_labels_validity(validation_labels)

	model = gamornet_build_model_tflearn(input_shape=input_shape)

	sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)
	model.compile(loss=loss, optimizer=sgd, metrics=['accuracy']) 


	callbacks_list = []
	
	if checkpoint_freq != 0:
		checkpoint = ModelCheckpoint(files_save_path + 'model_{epoch:02d}.hdf5', monitor='val_loss', verbose=verbose, save_best_only=False, save_weights_only=False, mode='auto', period=checkpoint_freq)
		callbacks_list.append(checkpoint)

	csv_logger = CSVLogger(files_save_path + "metrics.csv", separator=',', append=False)
	callbacks_list.append(csv_logger)
	
	if load_model == True:
		model = gamornet_load_model_tflearn(model,model_load_path)

	model.fit(training_imgs, training_labels, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(validation_imgs,validation_labels), shuffle=True, callbacks=callbacks_list)

	if save_model == True:
		model.save(files_save_path + "trained_model.hdf5")

	return model


	
def gamornet_tl_tflearn(training_imgs,training_labels,validation_imgs,validation_labels,input_shape,load_layers_bools = [True]*8,trainable_bools = [True]*8,model_load_path="./",files_save_path="./",epochs=100,checkpoint_freq=0,batch_size=64,lr=0.00001,momentum=0.9,decay=0.0,nesterov=False,loss='categorical_crossentropy',save_model=True,verbose=1):

	check_imgs_validity(training_imgs)
	check_imgs_validity(validation_imgs)
	check_labels_validity(training_labels)
	check_labels_validity(validation_labels)
	check_bools_validity(load_layers_bools)
	check_bools_validity(trainable_bools)

	model = gamornet_build_model_tflearn(input_shape=input_shape)
	model_new = clone_model(model)
	model = gamornet_load_model_tflearn(model,model_load_path)

	#Reversing the Order of the Bools because I will call .pop() on these later	
	load_layers_bools.reverse()
	trainable_bools.reverse()

	for i in range(len(model_new.layers)):
    
		if model_new.layers[i].count_params() != 0:
        
			model_new.layers[i].trainable = trainable_bools.pop()
        
			if load_layers_bools.pop() == True:
				model_new.layers[i].set_weights(model.layers[i].get_weights())
				print("Loading Layer" + str(i) + " from previous model.")
			else:
				print("Initializing Layer" + str(i)+ " from scratch")
            
		else:
			model_new.layers[i].set_weights(model.layers[i].get_weights())

	sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)
	model_new.compile(loss=loss, optimizer=sgd, metrics=['accuracy']) 

	callbacks_list = []
	
	if checkpoint_freq != 0:
		checkpoint = ModelCheckpoint(files_save_path + 'model_{epoch:02d}.hdf5', monitor='val_loss', verbose=verbose, save_best_only=False, save_weights_only=False, mode='auto', period=checkpoint_freq)
		callbacks_list.append(checkpoint)

	csv_logger = CSVLogger(files_save_path + "metrics.csv", separator=',', append=False)
	callbacks_list.append(csv_logger)
	

	model_new.fit(training_imgs, training_labels, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(validation_imgs,validation_labels), shuffle=True, callbacks=callbacks_list)

	if save_model == True:
		model_new.save(files_save_path + "trained_model.hdf5")

	return model_new


###########################################
###########################################
