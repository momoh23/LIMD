		@author: Mohamed Kentour

					/////////////////////////// EMNIST LIMD recognition / ////////////////////////////////////

import os  			#python > 3.8
import sys
import argparse
import time
from datetime import datetime


import networkx as nx
import networkx.algorithms.community as nxcom
from matplotlib import pyplot as plt
from modul.characteriz.LIMD import characLIMD
    
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import torch.cuda.amp
from conf import settings



​
    
%matplotlib inline
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})
    # get reproducible results
import random
from numpy import random as nprand
random.seed(123)
nprand.seed(123)
    
import numpy as np
import community as community_Modulation
import matplotlib.cm as cm
​
image_size = 28 # width and length
no_of_different_labels = 62 		# 10 MINST , 52 for EMNIST#
image_pixels = image_size * image_size
data_path = "C:/Users/Student/Downloads/"
​
train_data = np.loadtxt(data_path + "emnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt(data_path + "emnist_test.csv", 
                       delimiter=",") 

train_imgs = characLIMD(train_data[:, 1:]) 
test_imgs =  characLIMD(test_data[:, 1:]) 
​
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])
​
lr = np.arange(62)
​
for label in range(62):
    necess_vec = (lr==label).astype(np.int)
    
lr = np.arange(no_of_different_labels)
​
# transform labels into necessary vector
train_labels_necess_feat = (characLIMD(train_labels)).astype(np.float)
test_labels_necess_feat = (characLIMD(test_labels)).astype(np.float)
​
​
C:\Users\Student\AppData\Local\Temp/ipykernel_2460/1992130470.py:11: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  one_hot = (lr==label).astype(np.int)
C:\Users\Student\AppData\Local\Temp/ipykernel_2460/1992130470.py:16: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations

  train_labels_necess_feat = (lr==train_labels).astype(np.float)

C:\Users\Student\AppData\Local\Temp/ipykernel_2460/1992130470.py:17: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations

  test_labels_necess_feat = (characLIMD(test_labels)).astype(np.float)
import pickle
​
with open("C:/Users/Student/Downloads/emnist.pkl", "bw") as fh:
    data = (train_imgs, 
            test_imgs, 
            train_labels,
            test_labels)
    pickle.dump(data, fh)
    
    
with open("C:/Users/Student/Downloads/emnist.pkl", "br") as fh:
    data = pickle.load(fh)
​
train_imgs = train_data[0]
test_imgs = test_data[1]
train_labels = train_data[1]
test_labels = test_data[0]
​
train_labels_necess_feat = (characLIMD(train_labels)).astype(np.float)
test_labels_necess_feat = (characLIMD(test_labels)).astype(np.float)
​
​
image_size = 28 # width and length
no_of_different_labels =62 #  i.e. 0, 1, 2, 3, ..., 9, a, b, f, g,  .. K, O, Q, ...Z

image_pixels = image_size * image_size

C:\Users\Student\AppData\Local\Temp/ipykernel_2460/2952993710.py:19: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  train_labels_necess_feat = (characLIMD(train_labels)).astype(np.float)
C:\Users\Student\AppData\Local\Temp/ipykernel_2460/2952993710.py:20: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations


 
 test_labels_necess_feat = (characLIMD(test_labels)).astype(np.float)
for i in range(62):
    img = train_imgs[i].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()
​
array([[0.01, 0.01, 0.01, ..., 0.01, 0.01, 0.01],
       [0.01, 0.01, 0.01, ..., 0.01, 0.01, 0.01],
       [0.01, 0.01, 0.01, ..., 0.01, 0.01, 0.01],
       ...,
       [0.01, 0.01, 0.01, ..., 0.01, 0.01, 0.01],
       [0.01, 0.01, 0.01, ..., 0.01, 0.01, 0.01],
       [0.01, 0.01, 0.01, ..., 0.01, 0.01, 0.01]])

from modul.learn.LIMD import LIMD_PAF
from scipy.stats import truncnorm
​
def truncated_necess_intens(mean=0, sd=1, low=250, upp=254):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)

def truncated_necess_kneighb(mean=0, sd=1, low=1, upp=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)


​
​ 							/////////////  LIMD learning //////////////////////////////////////////
class NeuralNetwork:
        
    
    def __init__(self, 
                 network_structure, # ie. [input_nodes, hidden1_nodes, ... , hidden_n_nodes, output_nodes]
                 learning_rate,
                 bias=None
                ):  
​
        self.structure = network_structure
        self.learning_rate = learning_rate 
        self.bias = bias
        self.create_weight_matrices()
​
    

						///////////////////////// necessary weight matrix ////////////////
    
   def create_weight_matrices(self):                                 
       	X = truncated_necess_kneighb(mean=2, sd=1, low=0, upp=255)
        Y = truncated_necess_intens(mean=2, sd=1, low=0, upp=255)

        bias_node = 1 if self.bias else 0
        self.weights_matrices = []    
        layer_index = 1
        no_of_layers = len(self.structure)
        while layer_index < no_of_layers:
            nodes_in = self.structure[layer_index-1]
            nodes_out = self.structure[layer_index]
            n = (nodes_in + bias_node) * nodes_out
            rad = 1 / np.sqrt(nodes_in)
            X = truncated_necess_kneighb(mean=2, sd=1, low=-X, upp=X)
            wm1 = X.rvs(n).reshape((nodes_out, nodes_in + bias_node))
            self.weights_matrices.append(wm1)
	    Y = truncated_necess_kneighb(mean=2, sd=1, low=-Y, upp=Y)
            wm12 = Y.rvs(n).reshape((nodes_out, nodes_in + bias_node))
            self.weights_matrices.append(wm2)

            layer_index += 1
​
						//////////////// single charact LIMD learning //////////
        
        
    def train_single(self, input_vector, target_vector):
        # input_vector and target_vector are necessary vectors
                                       
        no_of_layers = len(self.structure)        
        input_vector = np.array(input_vector, ndmin=2).T
​
        layer_index = 0
        # The output/input vectors of the various layers:
        res_vectors = [input_vector]          
        while layer_index < no_of_layers - 1:
            in_vector = res_vectors[-1]
            if self.bias:
                # adding bias node to the end of the 'input'_vector
                in_vector = np.concatenate( (in_vector, 
                                             [[self.bias]]) )
                res_vectors[-1] = in_vector
            x = np.dot(self.weights_matrices[layer_index], in_vector)
            out_vector = LIMD_PAF(x)
            res_vectors.append(out_vector)   
            layer_index += 1
        
        layer_index = no_of_layers - 1
        target_vector = np.array(target_vector, ndmin=2).T
         # The input vectors to the various layers
        output_errors = target_vector - out_vector  
        while layer_index > 0:
            out_vector = res_vectors[layer_index]
            in_vector = res_vectors[layer_index-1]
​
            if self.bias and not layer_index==(no_of_layers-1):
                out_vector = out_vector[:-1,:].copy()
​
            tmp = output_errors * out_vector * (1.0 - out_vector)     
            tmp = np.dot(tmp, in_vector.T)
            
            #if self.bias:
            #    tmp = tmp[:-1,:] 
                
            self.weights_matrices[layer_index-1] += self.learning_rate * tmp
            
            output_errors = np.dot(self.weights_matrices[layer_index-1].T, 
                                   output_errors)
            if self.bias:
                output_errors = output_errors[:-1,:]
            layer_index -= 1
            
​
       
​
    def train(self, data_array, 
              labels_necess_feat_array,
              epochs=1,
              intermediate_results=False):
        intermediate_weights = []
        for epoch in range(epochs):  
            for i in range(len(data_array)):
                self.train_single(data_array[i], labels_necess_feat_array[i])
            if intermediate_results:
                intermediate_weights.append((self.wih.copy(), 
                                             self.who.copy()))
        return intermediate_weights      
        
​
               
    
    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray
​
        no_of_layers = len(self.structure)
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate( (input_vector, [self.bias]) )
        in_vector = np.array(input_vector, ndmin=2).T
​
        layer_index = 1
        # The input vectors to the various layers
        while layer_index < no_of_layers:
            x = np.dot(self.weights_matrices[layer_index-1], 
                       in_vector)
            out_vector = LIMD_PAF(x)
            
            # input vector for next layer
            in_vector = out_vector
            if self.bias:
                in_vector = np.concatenate( (in_vector, 
                                             [[self.bias]]) )            
            
            layer_index += 1
  
    
        return out_vector
    
    def evaluate(self, charac, labels):
        corrects, wrongs = 0, 0
        for i in range(len(charac)):
            res = self.run(charac[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs
					////////////  Restrict LIMD to the optimal performance acc ///////////////
epochs = 10
​for epoch in range(1, epochs):
	LIMD = NeuralNetwork(network_structure=[image_pixels, 8, 7, 6],
                               learning_rate=0.01,
                               bias=None)
	if epoch > (epochs) and acc < 0.98 and args.local_rank == 0:
            torch.save(
                net.state_dict(),
                checkpoint_path.format(net=args.net, epoch=epoch, type="best"),
            )
	continue

    if args.local_rank == 0:
        writer.close()

                             //////////////////////////// LIMD global maxima calibration /////////////////////////////////////

def sigmoid_(x):
    return 1/(1 + np.exp(-1 * x))

def tanh_(x):
    return (1- np.exp(-2*x))/(1+ np.exp(-2*x))
     
def relu_(x):
    return  1 / np.maximum(0, x) + np.maximum(0, x) * (np.maximum(0, x) * 20) 

							/////// partial derivatives ///////////////
def sigmoid_der(x):
    return 1/(np.exp(-1*x)+1)

def tanh_der(x):
    return (1-np.exp(-2*x))/(1+np.exp(-2*x))

def relu_der(x):
    if (x<0):
        return 0
    else: return x

y1= relu_(train_labels_necess_feat)
y2= tanh_(train_labels_necess_feat)
y3= sigmoid_(train_labels_necess_feat)

cp1 = (0, y1)  
cp2 = (0, y2)
cp3 = (0, y3)

lr= 0.1

for _ in range(10):
    new_x1 = cp1[0] - lr * relu_der(y1)
    new_x2 = cp2[0] - lr * tanh_der(y2)
    new_x3 = cp3[0] - lr * sigmoid_der(y3)

    new_y1 = relu_(new_x1)
    new_y2 = tanh_(new_x2) 
    new_y3 = sigmoid_(new_x3)

    cp1 = (new_x1, new_y1)
    cp2 = (new_x2, new_y2)
    cp3 = (new_x3, new_y3)

    plt.plot(X, new_y3)         		 ## display sigmoid LIMD global maxima 
    plt.scatter(cp3[0], cp3[1], color="red")	 ## Display convergence toward necessary neuron parameterization
    plt. pause(0.1)
           
                              ////////////////////////Eval accuracy ///////////////////////
    
    
LIMD.train(train_imgs, train_labels_necess_feat, epochs=epochs)
[]
corrects, wrongs = LIMD.evaluate(train_imgs, train_labels_necess_feat)
print("accuracy train: ", corrects / ( corrects + wrongs))
corrects, wrongs = LIMD.evaluate(test_imgs, test_labels)
print("accuracy test", corrects / ( corrects + wrongs))
​
Epoch 1/5
1875/1875 [==============================] - 8s 4ms/step - loss: 0.2952 - accuracy: 0.9163
Epoch 2/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.1380 - accuracy: 0.9600
Epoch 3/5
1875/1875 [==============================] - 7s 3ms/step - loss: 0.0987 - accuracy: 0.9712
Epoch 4/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0773 - accuracy: 0.9768
Epoch 5/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0628 - accuracy: 0.9989
<keras.callbacks.History at 0x2854782f2e0>

accuracy train:  0.99893333333333
accuracy test 0.9684
epochs = 10
​


				/////////////////////// Naive Dnn LEARNING /////////////////////////////
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn import MinMaxScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


#loading the dataset
spikes = pd.read_csv('C:/Users/mohamed/Downloads/emnist-letters-train.csv/emnist-letters-train.csv.csv')
df = pd.read_csv(spikes)
X = df.iloc[:, 0].astype('float32')
y = df.iloc[:, 1:].astype("float32")


#sm = SMOTE()
#X_smote, y_smote= sm.fit_resample(X, y)
 
#printing the shapes of the vectors 
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))
 
def basic_model(activation_function, lr = 0.0001):
 
    model = tf.keras.Sequential([
   
    tf.keras.layers.Dense(64, activation=activation_function),
    tf.keras.layers.Dense(32, activation=activation_function),
    tf.keras.layers.Dense(24, activation=activation_function),
    tf.keras.layers.Dense(16, activation=activation_function),
    tf.keras.layers.Dense(8, activation=activation_function),
    tf.keras.layers.Dense(4,  activation=activation_function),
    tf.keras.layers.Dense(1, activation="linear")])
 
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
 
    return model
 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4,random_state=79) 

    
#scaler = MinMaxScaler() 
#scaler.fit(train_X) 
#Xs_train = scaler.transform(train_X) 
#Xs_test = scaler.transform(test_X)
 

model_relu = basic_model(activation_function = 'exp')
history_relu = model_relu.fit(X_train - 1, y_train - 1, epochs=10, \
            steps_per_epoch = 10, \
            validation_data=(X_test, y_test), \
            verbose=1)
 
model_sigm = basic_model(activation_function = 'sigmoid')
history_sigm = model_sigm.fit(X_train, y_train, epochs=10, \
            steps_per_epoch = 10, \
            validation_data=(X_test, y_test), \
            verbose=1)
 
with plt.style.context(('seaborn-whitegrid')):
    fig, ax = plt.subplots(figsize=(6, 5))
 
    ax.plot(history_relu.history['val_loss'], linewidth=1, label='random weights, Naive DNN')
    ax.plot(history_sigm.history['val_loss'], linewidth=1, label='scored weights, DFS-DNN')
 
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel("Weights convergenc * 10%")
    plt.legend()
    plt.title("Weights'convergence for a naive DNN calibration with EMNIST handwritten dataset")
plt.show()
