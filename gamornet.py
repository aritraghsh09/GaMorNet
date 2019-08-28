##################################################
#  gamornet.py
#  This file is simply meant to show the TFLearn functions we use to build GaMorNet. 
#  The commented numbers refer to layer numbers in Table. 3 of the paper
#  THIS FILE IS NOT MEAN TO BE RUN WITHOUT MODIFICATIONS. PLEASE REFER TO THE README OF THIS REPOSITROY FOR MORE INFO
##################################################

###### Start of building GaMorNet #######

network = input_data(shape=[None, 167, 167, 1])    #Layer 1
#Since we do training in batches, the first None serves as to empower us to choose any batch size that we may want.
#The next three numbers represent the shape of the input images. We use 167*167 cutouts with a single channel.


network = conv_2d(network, 96, 11, strides=4, activation='relu') #Layer 2
network = max_pool_2d(network, 3, strides=2) #Layer 3

network = local_response_normalization(network) #Layer 4

network = conv_2d(network, 256, 5, activation='relu') #Layer 5
network = max_pool_2d(network, 3, strides=2) #Layer 6

network = local_response_normalization(network) #Layer 7

network = conv_2d(network, 384, 3, activation='relu') #Layer 8
network = conv_2d(network, 384, 3, activation='relu') #Layer 9

network = conv_2d(network, 256, 3, activation='relu') #Layer 10
network = max_pool_2d(network, 3, strides=2) #Layer 11

network = local_response_normalization(network)  #Layer 12

network = fully_connected(network, 4096, activation='tanh') #Layer 13
network = dropout(network, 0.5) #Layer 14

network = fully_connected(network, 4096, activation='tanh') #Layer 15
network = dropout(network, 0.5) #Layer 16

network = fully_connected(network, 3, activation='softmax') #Layer16 -- This is the output layer

##### End of Defining GaMorNet ############

#training method and parameters 
network = regression(network, optimizer='momentum',loss='categorical_crossentropy',learning_rate=0.0001) 
#The Learning Rate parameter greatly affects the results. 

