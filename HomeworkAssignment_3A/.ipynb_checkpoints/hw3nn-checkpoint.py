# -*- coding: utf-8 -*-


import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import accuracy_score

class Layer():
    input_dim = None
    
    def __init__(self,neurons,activation,input_dim=None):
        np.random.seed(9)
    
        self.neurons = neurons
        
        if activation == 'relu':
            self.activation_fun = relu
            self.derivative_act = derivative_relu
            
        elif activation == 'sigmoid':
            self.activation_fun = sigmoid
            self.derivative_act = derivative_sigmoid
            
        if input_dim is not None:
            self.input_dim = input_dim
            self.weights = np.asarray([ np.random.rand(input_dim) for x in range(neurons)])
            self.biases = np.asarray([np.zeros(1) for x in range(neurons)])
        
        
    def __str__(self):
        return 'Hidden Layer with '+ str(self.neurons)+' neuros'
    
class MLP():
    np.random.seed(9)
    cost_h = []
    layers = []  #store all layers
    z_array = [] #store all weighted sums  per layer
    a_array = [] #store all activation units per layer a = act_fun(z)
    
    def __init__(self):
        pass
    
    def add(self,Layer):
        if Layer.input_dim is None: #infer input dimension 
            Layer.input_dim = self.layers[-1].neurons
            Layer.weights = np.asarray([ np.random.randn(Layer.input_dim) for x in range(Layer.neurons)])
            Layer.biases = np.asarray([ np.zeros(1) for x in range(Layer.neurons)])
        self.layers.append(Layer)
      
    
    #find final output of the network and save all activations units and zetas
    def feedforward(self,a): 
        
        
        self.z_array.clear()
        self.a_array.clear()
        #self.a_array.append(a)
        
        self.a_array.append(a)
        for i in range(len(self.layers)):
            z = np.dot(self.layers[i].weights,a) + self.layers[i].biases
            self.z_array.append(z) 
            a = self.layers[i].activation_fun(z)
          #  a[a<0.001] = 0.5
            self.a_array.append(a)
        return a
    
    #once you computed your output you have to find how good the result was
    
    def backpropagation(self,x,y): #not working
        grad_b = [np.zeros(layer.biases.shape) for layer in self.layers] #dC/db
        grad_w = [np.zeros(layer.weights.shape) for layer in self.layers] #dC/dw
        
        
        #how the cost function is behaving ?
        dr_cross =  derivative_bCrossEntroypy(x,y)
        dr_actfun = self.layers[-1].derivative_act(self.z_array[-1])
        dC_do = dr_cross*dr_actfun
        
            
        #dC_do = der_mse(self.a_array[-1],y)*self.layers[-1].derivative_act(self.z_array[-1])
        
        grad_b[-1] = dC_do
        grad_w[-1] = dC_do.dot(self.a_array[-2].T) #errors second-last layer(10 outputs) #for each weight
        
    
        for l in range(len(self.layers)-1,0,-1): #start from the last output layer and propagate the error
            #print(l,'ok')
            #chain rule -> d^l = w^(l+1).T @ d^l+1 * derivate_actfun(z^l)
            
            dC_do = self.layers[l].weights.T.dot(dC_do)*self.layers[l].derivative_act(self.z_array[l-1])
            #chain rule
        
            grad_b[l-1] = dC_do
            grad_w[l-1] = dC_do.dot(self.a_array[l-1].T)
        
        return (grad_b,grad_w)
    
    
    def train(self,x_train,y_train,batch_size,epochs, lr):
        
        #x_train_norm = x_train
        x_train_norm = MinMaxScaler().fit_transform(x_train) #you need to scale data otherwise u are gonna get gradient exploding/zeros at denominatore
        
        train = np.concatenate((x_train_norm,y_train[:,np.newaxis]),axis=1) #to perform batch grdient descent 
        
        for i in range(epochs):
            print('Epoch {0}'.format(i))
            #shuffle train set for batch gradient descent
            
            np.random.shuffle(train)
            
            #generate batches
            batches = np.asarray([train[i:i+batch_size] for i in  range(0,train.shape[0],batch_size)])
            
            #apply GD for each batch
            for batch in batches:
                self.update_weights(batch,lr)
    
    def predict(self,x_test):        
        
        return self.feedforward(x_test.T)
        
        
        
        
        
    
    def update_weights(self,batch,lr):
         
         sum_=0
        
         grad_b_sum = [np.zeros(layer.biases.shape) for layer in self.layers] #dC/db
         grad_w_sum = [np.zeros(layer.weights.shape) for layer in self.layers]
         
         batch_len = batch.shape[0]         
         
         for b in range(batch_len):
            
           
            x = np.expand_dims(batch[b][:-1],axis=1)
            
            y = np.expand_dims(batch[b][-1],axis=1)
             
            
            output = self.feedforward(x) #forward value over the network
            binary_crossEntropy(output,y)
            
            
            grad_b, grad_w = self.backpropagation(output,y) #use backpropagation to compute dC/dW
            
            #sum over all observations in the batch
            grad_b_sum = [nb+dnb for nb, dnb in zip(grad_b, grad_b_sum)]
            grad_w_sum = [nw+dnw for nw, dnw in zip(grad_w, grad_w_sum)]
#            for i in range(len(grad_w_sum)):
#                grad_w_sum[i]+= grad_w[i]
#            for i in range(len(grad_b_sum)):
#                grad_b_sum[i]+= grad_b[i]
        
            
         self.cost_h.append(sum_/batch_len)
        
            
         new_w = [layer.weights - (lr/batch_len)*gw for layer,gw in zip(self.layers,grad_w_sum)]
         new_b = [layer.biases - (lr/batch_len)*bw for layer,bw in zip(self.layers,grad_b_sum)]
         
         for i in range(len(self.layers)):
             self.layers[i].weigths = new_w[i]
             self.layers[i].biases  = new_b[i]
         
    
            
            

    def __str__(self):
        print(self.layers)
        
    def __del__(self):
        self.layers.clear()
        self.a_array.clear()
        self.z_array.clear()
        
    def print_layers(self):
        for layer in self.layers:
            print(layer)


def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def binary_crossEntropy(o, y):#o: output ,y:target 
    return -(y*np.log(o+1e-9) + (1-y)*np.log(1-o+1e-9))

def derivative_bCrossEntroypy(o,y):
    return -(y/(o + 1e-9)+(1-y)/(1-(o+1e-9)))
             
    
def relu(x):
    return np.maximum(0,x)

def reluN(x, alpha=0., max_value=None, threshold=0.):
    if max_value is None:
        max_value = np.inf
    above_threshold = x * (x >= threshold)
    above_threshold = np.clip(above_threshold, 0.0, max_value)
    below_threshold = alpha * (x - threshold) * (x < threshold)
    return below_threshold + above_threshold

def binary_crossentropyN(target, output, from_logits=False):
    if not from_logits:
        output = np.clip(output, 1e-7, 1 - 1e-7)
        output = np.log(output / (1 - output))
    return (target * -np.log(sigmoid(output)) +
            (1 - target) * -np.log(1 - sigmoid(output)))
    
    
def derivative_relu(x):
   # x = x.flatten()
    der = np.zeros(x.shape)
    der[x>0] = 1
    return der

if __name__=='__main__':
    
    df = pd.read_excel('/Users/andreafavia/Desktop/EIT/TU:e/DataMining/HW_NN/HW3Atrain.xlsx')
    df_test = pd.read_excel('/Users/andreafavia/Desktop/EIT/TU:e/DataMining/HW_NN/HW3Avalidate.xlsx')
    
    x_test = df_test.iloc[:,:-1].values
    y_test = df_test.iloc[:,-1].values
    x_train = df.iloc[:,:-1].values
    y_train  = df.iloc[:,-1].values
    
 
    
    mlp = MLP()
    mlp.add(Layer(10,input_dim=2,activation='relu'))
    mlp.add(Layer(10,activation='relu'))
    mlp.add(Layer(1,activation='sigmoid'))
    
    mlp.train(x_train,y_train,32,100,0.1)

    pred = mlp.predict(x_test)
    