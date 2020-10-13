#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# Dense

# Define MLP class
# #binary classification problem 

# In[10]:


class Layer():
    input_dim = None
    
    def __init__(self,neurons,activation,input_dim=None):
    
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
            self.biases = np.asarray([np.random.randn(1) for x in range(neurons)])
        
        
    def __str__(self):
        return 'Hidden Layer with '+ str(self.neurons)+' neuros'


# In[ ]:


#prova
inputd = 2
x = np.array([1,2])
weights =np.asarray([ np.random.rand(inputd) for x in range(10)])
biases = np.asarray([np.random.randn() for x in range(10)])
weights @ x + biases


# In[36]:


class MLP():
    
    layers = []  #store all layers
    z_array = [] #store all weighted sums  per layer
    a_array = [] #store all activation units per layer a = act_fun(z)
    
    def __init__(self):
        pass
    
    def add(self,Layer):
        if Layer.input_dim is None: #infer input dimension 
            Layer.input_dim = self.layers[-1].neurons
            Layer.weights = np.asarray([ np.random.randn(Layer.input_dim) for x in range(Layer.neurons)])
            Layer.biases = np.asarray([ np.random.randn(1) for x in range(Layer.neurons)])
        self.layers.append(Layer)
      
    #I use sgd to train the model 
    def train(self,inizialization,l_rate=0.1,batch_size=32):
        #for epoch in #epochs
            #batches <-- divide in batches
            #for batch in batches
                #forward propagate
                #backpropagation of error
                #update weights     
        pass
    
    
    #find final output of the network and save all activations units and zetas
    def feedforward(self,a): 
        #must inizialize these arrays?!
        self.z_array.clear()
        self.a_array.clear()
        #self.a_array.append(a)
        
        a = a.T
        for i in range(len(self.layers)):
            z = np.dot(self.layers[i].weights,a) + self.layers[i].biases
            self.z_array.append(z) 
            a = self.layers[i].activation_fun(z)
            self.a_array.append(a)
        return a
    
    #once you computed your output you have to find how good the result was
    
    def backpropagation(self,x,y): #not working
        grad_b = [np.zeros(layer.biases.shape) for layer in self.layers] #dC/db
        grad_w = [np.zeros(layer.weights.shape) for layer in self.layers] #dC/dw
        
        
        #how the cost function is behaving ?
        dr_cross =  derivative_bCrossEntroypy(self.a_array[-1],y)
        dr_actfun = self.layers[-1].derivative_act(self.z_array[-1])
        dC_do = dr_cross*dr_actfun
        
        

        #print(self.a_array[-2].T)
        grad_b[-1] = dC_do
        grad_w[-1] = dC_do.dot(self.a_array[-2].T) #errors second-last layer(10 outputs) #for each weight
    
        for l in range(len(self.layers)-1,0,-1): #start from the last output layer and propagate the error
            print(l,'ok')
            #chain rule -> d^l = w^(l+1).T @ d^l+1 * derivate_actfun(z^l)
            
            dC_do = self.layers[l].weights.T.dot(dC_do)*self.layers[l].derivative_act(self.z_array[l-1])
            #chain rule
        
            grad_b[l-1] = dC_do
            grad_w[l-1] = dC_do.T.dot(self.a_array[l-1])
        
        return (grad_b,grad_w)
            
    
    def __str__(self):
        print(self.layers)
        
    def __del__(self):
        self.layers.clear()
        self.a_array.clear()
        self.z_array.clear()
        
    def print_layers(self):
        for layer in self.layers:
            print(layer)
                 


# In[ ]:


dC_do = derivative_bCrossEntroypy(0.9,1)*derivative_sigmoid(mlp.z_array[-1]) 
for l in range(1,-1,-1): #start from the second-last output layer and propagate the error
        print(l)
        dC_do = mlp.layers[l+1].weights.T.dot(dC_do)* mlp.layers[l].derivative_act(mlp.z_array[l])
        dC_dw = mlp.a_array[l-1].dot(dC_do.T)
#first layer
dC_do = mlp.layers[0].weights.T.dot(dC_do)*mlp.layers[l].derivative_act(mlp.z_array[l])


# In[35]:


del mlp


# In[37]:


mlp = MLP()
mlp.add(Layer(10,input_dim=2,activation='relu'))
mlp.add(Layer(10,activation='relu'))
mlp.add(Layer(1,activation='sigmoid'))


# In[38]:


[layer.weights.shape for layer in mlp.layers]


# In[15]:


[layer.biases.shape for layer in mlp.layers]


# In[39]:


a = np.random.randn(1,2)
res = mlp.feedforward(a)


# In[42]:


err = binary_crossEntropy(res,0)
err


# In[43]:


mlp.z_array


# In[44]:


deltaL = derivative_bCrossEntroypy(res,0)*derivative_sigmoid(mlp.z_array[-1])


# In[79]:


deltaL


# In[47]:


dWL = mlp.a_array[-2]*deltaL
dWL #for each weight i gotta removen


# In[83]:


deltal2  =(mlp.layers[-1].weights.T*deltaL)*derivative_relu(mlp.z_array[-2])
deltal2


# In[62]:


mlp.a_array


# In[ ]:


mlp.a_array
dLout = derivative_bCrossEntroypy(0.9,1)*derivative_sigmoid(mlp.z_array[-1])
dwout = mlp.a_array[-2].dot(dLout)


dL2 = mlp.layers[-1].weights.T.dot(dLout)*derivative_relu(mlp.z_array[-2])
dC_Dw = dL2.dot(mlp.a_array[-3])

dC_Dw


# ### Activation and Loss functions.
# #####  I use Sigmoid as activation function for the hidden layers. For the output layer I use a sigmoid activation function. As  loss function I use binary cross entropy. The choice to use sigmoid as output activation function is due to the fact that  we are dealing with a binary classification problem and the sigmoid function will give us a value in the range (0,1).

# In[5]:


def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


# In[6]:


def relu(x):
    return np.maximum(0,x)

def derivative_relu(x):
   # x = x.flatten()
    der = np.zeros(x.shape)
    der[x>0] = 1
    return der


# In[8]:


def binary_crossEntropy(o, y):#o: output ,y:target 
    return -(y*np.log(o) + (1-y)*np.log(1-o))
def derivative_bCrossEntroypy(o,y):
    return -(y/o+ (1-y)/(1-o))


# In[ ]:




