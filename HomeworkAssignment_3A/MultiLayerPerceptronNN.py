#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:08:50 2019

@author: andreafavia
"""

##to do:
# cost on validation (done)
# set and plot it (done)
#early stopping and adaptive learning rate(done)

#might do regularization

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import  accuracy_score,confusion_matrix

def lrelu(x,deriv=False):
    if deriv:
        der = np.ones(x.shape)*0.01
        der[x>0] = 1
        return der
    return (x*(x>0)+0.01*x*(x<=0))

def sigmoid(z,deriv=False):
    if deriv:
        return sigmoid(z)*(1-sigmoid(z))
    return (1/(1+np.exp(-z)))


def relu(x,deriv=False):
    if deriv:
        der = np.zeros(x.shape)
        der[x>0] = 1
        return der
    
    return x*(x>=0)


class MLP():
    
    np.random.seed(1) #use a random seed
    
    n_epochs = 0
    
    actf =            []    
    activations =     []
    cost =            []
    cost_validation = []
    zetas =           []
    shapes =          []
    
    def __init__(self,nn_architecture):    
        
        ''' shapes array of tuple (ouput,input) for each layer'''
        
        if len(nn_architecture['layers']) != len(nn_architecture['activations']):
            raise ValueError('For each layer there must be an activation function')
            
        self.shapes = nn_architecture['layers']
        self.actf = nn_architecture['activations']
        #Kaiming weights normalization
        self.weights = np.asarray([np.random.normal(0,np.sqrt(2/shape[0]),size=shape) for shape in self.shapes])
        #self.weights = np.asarray([np.random.randn(*shape)*np.sqrt(2/shape[0]) for shape in self.shapes])
        self.biases = np.asarray([np.zeros((shape[0],1)) for shape in self.shapes])  #biases are 0 at the beginning
        
    
    def feedforward(self,a): #feedforward
        
        self.activations.clear()
        self.zetas.clear()
        
        self.activations.append(a)
        
        for i in range(len(self.shapes)):
            #print(len(self.shapes))
          
            z = self.weights[i] @ a + self.biases[i] 
            self.zetas.append(z)
            a = self.actf[i](z)
            self.activations.append(a)
            
        return a #return sigmoid(last_layer_value)
    
    def backprop(self,o,y):
        
        grad_w =np.asarray([np.zeros(weight.shape) for weight in self.weights])
        grad_b =np.asarray([np.zeros(bias.shape) for bias in self.biases])
        
        #use binary cross entropy -> d^L = grad_a(Cost)*sig'(z^L)
        
        dC_do = mean_squared_error(o,y,deriv=True)* sigmoid(self.zetas[-1],deriv=True) # calculate delta_lastLayer 
        
        grad_b[-1] = dC_do
        grad_w[-1] = dC_do @ self.activations[-2].T #
        
        #backpropagate error
        
        for l in range(2,len(self.shapes)+1):
           
             #chain rule -> d^l = w^(l+1).T @ d^l+1 * derivate_actfun(z^l)
            
            dC_do = np.multiply((self.weights[-l+1].T @ dC_do), self.actf[-l](self.zetas[-l],deriv=True))

            #compute grad bias and grad weights
            grad_b[-l] = dC_do
            grad_w[-l] = dC_do @ self.activations[-l-1].T
            
        return (grad_b,grad_w)
    
    
    def train(self,df_train,df_test,lr=0.3,batch_size=32,epochs=100,adapt_lr = True, early_stopping = 5):
        
        last_cost = np.iinfo('int32').max #max value to compare with
        no_impr = 0
        
        x_test_norm =  MinMaxScaler().fit_transform(df_test.values[:,:-1]) 
        y_test = df_test.values[:,-1].reshape(-1,1)
        
        x_train_norm = MinMaxScaler().fit_transform(df_train.values[:,:-1])  #normalize data ( only features)[ even though MinMaxScaler won't affect y vector]
        y_train = df_train.values[:,-1].reshape(-1,1)
        df_train_norm = np.concatenate((x_train_norm,df_train.values[:,-1].reshape(-1,1)),axis=1) # rebuild dataframe
    
        
        for e in range(epochs):
            
            
            np.random.shuffle(df_train_norm) #shuffle the dataframe
        
            batches = [df_train_norm[bs:bs+batch_size] for bs in range(0,len(df_train_norm),batch_size) ] #generate batches 
            
            for batch in batches: #for each batch compute 
                
                #reduce learning rate of a magnitude after 130 epochs (adaptive lr)
                if adapt_lr and e > 130:
                    lr = lr/10
                    
                    
                self.update_wb(batch,lr)
#             
            
            cost = mean_squared_error(self.feedforward(x_train_norm.T),y_train)
            cost_validation = mean_squared_error(self.feedforward(x_test_norm.T),y_test) #my validation is the test set itself
            
            train_pred,y_train = self.predict(df_train)
            accuracy_train = accuracy_score(train_pred,y_train)
            
          #  print(cost,last_cost,cost-last_cost,sep='---')
          
             #Early stopping: if the cost in the validate sample ( in our case directly on the test set)
             #does not decrease for more than *early_stopping* epochs I may start overfitting the training set
            if cost_validation >= last_cost:
               no_impr+=1
            else:
               no_impr = 0
                
            if early_stopping != None and no_impr>= early_stopping:
               break
           
                
            last_cost = cost_validation
            
            self.cost.append(cost)
            self.cost_validation.append(cost_validation)
            self.n_epochs = e+1
            
            
            print('epoch {0}--> loss:  {1} -----> acc = {2}'.format(e,cost,accuracy_train))
            
                
                
    
    def accuracy_score(pred,y):
        return ((predictions == y_test).sum() / y_test.shape[0])
            
                
    def update_wb(self,batch,lr):
        
        
        
        grad_w_tot = np.asarray([np.zeros(weight.shape) for weight in self.weights])
        grad_b_tot = np.asarray([np.zeros(bias.shape) for bias in self.biases])
        
        x_train = np.expand_dims(batch[:,:-1],axis=1)
        y_train = np.expand_dims(batch[:,-1],axis=1)
        
        
        for x,y in zip(x_train,y_train): #for each sample i use forward and backprop to get gradients of weights/baises
            
            output = self.feedforward(x.T)
        
            gradb,gradw = self.backprop(output,y)
            
            #must sum grad_w/grad_b in the same batch 
            grad_w_tot = [gradw_i+gwt for gradw_i,gwt in zip(gradw,grad_w_tot)]
            grad_b_tot = [gradb_i+gbt for gradb_i,gbt in zip(gradb,grad_b_tot)]
        
        #update weights and biases --> w = w - lr/len(batch)*grad_w
        
        self.weights = [weight - (lr/len(batch))*gw_i for weight,gw_i in zip(self.weights,grad_w_tot)]
        self.biases = [bias - (lr/len(batch))*gb_i for bias,gb_i in zip(self.biases,grad_b_tot )]
        
      
    
    def predict(self,df_test):
       
        x_test = MinMaxScaler().fit_transform(df_test.values[:,:-1]) #normalize test set
        y_test = df_test.values[:,-1].reshape(-1,1) #(n_observations,1)
        
        
        predictions = []
        for x in x_test:
            x = x.reshape(-1,1)
            predictions.append( self.feedforward(x))
            
        predictions= np.asarray(predictions)
        predictions[predictions<0.5] = 0
        predictions[predictions>=0.5] =1 
        predictions = predictions.flatten().reshape(-1,1)
        return (predictions,y_test)
        
        


# calculate mean squared error
def mean_squared_error(actual, predicted,deriv=False):
    if deriv==True:
        return (actual-predicted)
    
    sum_square_error = 0.0
    for i in range(len(actual.T)):
        sum_square_error += (actual.T[i] - predicted[i])**2.0
    mean_square_error = 1.0 / len(actual) * sum_square_error
    return mean_square_error

     
        
if __name__=='__main__':
    
    
    df = pd.read_excel('HW3Atrain.xlsx')
    df_test = pd.read_excel('HW3Avalidate.xlsx')
    
    n_epochs = 200
    #del mlp
    nn_architecture = {
            'layers':[(10,2),(10,10),(1,10)],
            'activations':[relu,relu,sigmoid]
            }
    
    mlp = MLP(nn_architecture)
    mlp.train(df,df_test,epochs=n_epochs,batch_size=16,lr=0.3)
    
    predictions,y_test = mlp.predict(df_test)
    accuracy = accuracy_score(predictions,y_test)
    
    plt.figure(figsize = (8,5))
    plt.plot(range(mlp.n_epochs),mlp.cost)
    plt.plot(range(mlp.n_epochs),mlp.cost_validation,'r')
    plt.title('MSE error over epochs')
    plt.xlabel('number of epochs')
    plt.ylabel('error')
    plt.legend(['Error on Training','Error on Test'])
    
    print('accuracy: ',accuracy)
    
    df_cm = pd.DataFrame(confusion_matrix(predictions.flatten().reshape(-1,1),y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (8,5))
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm,annot=True,cmap='Blues')
    

            

    
