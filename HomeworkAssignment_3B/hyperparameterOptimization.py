#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group 58
Andrea Favia
Paolo Berizzi
Tristan Tomilin
Aletta Tordai
Eliza Starr
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.offline import plot
import plotly.graph_objects as go
from pandas.plotting import parallel_coordinates
from sklearn.metrics import  accuracy_score,confusion_matrix
import seaborn as sns

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
    
    def __init__(self,nn_architecture, mean=0, devStd=1, zeroWeights=False, zeroBiases=True):    
        
        if len(nn_architecture['layers']) != len(nn_architecture['activations']):
            raise ValueError('For each layer there must be an activation function')
            
        self.shapes = nn_architecture['layers']
        self.actf = nn_architecture['activations']
        #Kaiming weights normalization
        if(zeroWeights):
            self.weights = np.asarray([np.zeros((shape)) for shape in self.shapes])
        else:
            self.weights = np.asarray([np.random.normal(mean, devStd, shape) for shape in self.shapes])
            #self.weights = np.asarray([np.random.randn(*shape)*np.sqrt(2/shape[0]) for shape in self.shapes])
        
        if(zeroBiases):
            self.biases = np.asarray([np.zeros((shape[0],1)) for shape in self.shapes])  #biases are 0 at the beginning
        else:
            self.biases = np.asarray([np.random.normal(mean, devStd, (shape[0],1)) for shape in self.shapes])
            #self.biases = np.asarray([np.random.randn(shape[0],1) for shape in self.shapes])
    
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
    
    
    def train(self,df_train,df_test,epochs=100,batch_size=32,lr=0.3,adapt_lr = True, early_stopping = 5):
        
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
            
            
            #print('epoch {0}--> loss:  {1} -----> acc = {2}'.format(e,cost,accuracy_train))
            
                
                
    
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

def tryWithZero():
    n_epochs = 200
    b_size = 16
    l_rate = 0.3
    nn_architecture = {
            'layers':[(10,2),(10,10),(1,10)],
            'activations':[relu,relu,sigmoid]
            }
    mlp = MLP(nn_architecture, zeroWeights=True, zeroBiases=True)
    mlp.train(df_train,df_test, n_epochs, b_size, l_rate)
    
    predictions,y_test = mlp.predict(df_test)
    accuracy = accuracy_score(predictions,y_test)
     
    plt.plot(range(mlp.n_epochs),mlp.cost)
    plt.plot(range(mlp.n_epochs),mlp.cost_validation,'r')
    plt.title('MSE error over epochs')
    plt.xlabel('number of epochs')
    plt.ylabel('error')
    plt.legend(['Error on Training','Error on test'])
    
    print('accuracy: ',accuracy)
    confMat = confusion_matrix(predictions.flatten().reshape(-1,1),y_test)
    confMatDataframe = pd.DataFrame(confMat)
    confMatDataframe.index.name = 'Actual'
    confMatDataframe.columns.name = 'Predicted'
    sns.heatmap(confMatDataframe, cmap="Blues", annot=True)
    plt.show()
    
def heatmapAccuracy():
    n_epochs = 200
    b_size = 16
    nn_architecture = {
            'layers':[(10,2),(10,10),(1,10)],
            'activations':[relu,relu,sigmoid]
            }
    lrate_array = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    mean = 0
    stdDev_array = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    dataFrame = np.zeros((6, 11)) 
    for i in range (len(lrate_array)):
        for j in range (len(stdDev_array)):
            mlp = MLP(nn_architecture, mean, stdDev_array[j])
            mlp.train(df_train,df_test, n_epochs, b_size, lrate_array[i])
            predictions,y_test = mlp.predict(df_test)
            dataFrame[i][j] = accuracy_score(predictions,y_test)
    
    dataFrame = pd.DataFrame(dataFrame) 
    dataFrame.columns = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]
    dataFrame.columns.name = "Standard deviation"
    dataFrame.index = ["0.0001", "0.001", "0.01", "0.1", "1", "10"]
    dataFrame.index.name = "Learning rate"
    
    sns.heatmap(dataFrame, cmap="Blues", annot=True)
    plt.show()
    
def heatmapActivation(mlpToTest, df_test, title_x):
    x_test = MinMaxScaler().fit_transform(df_test.values[:,:-1]) #normalize test set
    y_test = df_test.values[:,-1].reshape(-1,1) #(n_observations,1)
        
    dataLayer1 = np.zeros((82, 10))
    dataLayer2 = np.zeros((82, 10))
    item = 0
    for x in x_test:
        x = x.reshape(-1,1)
        mlpToTest.feedforward(x)
        dataLayer1[item] = np.reshape(mlpToTest.activations[1], (10))
        dataLayer2[item] = np.reshape(mlpToTest.activations[2], (10))
        item += 1
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle(title_x, fontsize=16)
    ax1.set_title("First hidden layer")
    dataLayer1 = pd.DataFrame(dataLayer1)
    dataLayer1.index.name = "Point id"
    dataLayer1.columns.name = "Hidden neuron id"
    heatL1 = sns.heatmap(dataLayer1, ax=ax1, cmap="Greens")
    ax2.set_title("Second hidden layer")
    dataLayer2 = pd.DataFrame(dataLayer2)
    dataLayer2.index.name = "Point id"
    dataLayer2.columns.name = "Hidden neuron id"
    heatL2 = sns.heatmap(dataLayer2, ax=ax2, cmap="Oranges")
    fig.set_size_inches(12.0, 6.0)
    
    plt.show()
    return(heatL1, heatL2)

def differentActivations(df_train, df_test):
    n_epochs = 100
    mean = 0
    bSize = 16
    
    worstStdDev = 0.6
    worstLRate = 0.001
    
    bestStdDev = 0.9
    bestLRate = 1
    #definition of the architecture
    nn_architecture = {
            'layers':[(10,2),(10,10),(1,10)],
            'activations':[relu,relu,sigmoid]
            }
    
    mlpWorst = MLP(nn_architecture, mean, worstStdDev)
    mlpBest = MLP(nn_architecture, mean, bestStdDev)
    
    #initial heatmap (random)
    heatmapActivation(mlpWorst, df_test, "Worst model - Initial activations")
    heatmapActivation(mlpBest, df_test, "Best model - Initial activations")
    
    #Half training
    mlpWorst.train(df_train, df_test, n_epochs, bSize, worstLRate)
    mlpBest.train(df_train, df_test, n_epochs, bSize, bestLRate)
    
    #show heatmap halfaway
    heatmapActivation(mlpWorst, df_test, "Worst model - Halfway activations")
    heatmapActivation(mlpBest, df_test, "Best model - Halfway activations")
    
    #Second half training
    mlpWorst.train(df_train, df_test, n_epochs, bSize, worstLRate)
    mlpBest.train(df_train, df_test, n_epochs, bSize, bestLRate)
    
    #Show final heatmap
    heatmapActivation(mlpWorst, df_test, "Worst model - Ending activations")
    heatmapActivation(mlpBest, df_test, "Best model - Ending activations")

def hyperparamOptimization(df_train):
    
    widths_array = [10, 100]
    layers_array = [0, 1, 2, 7]
    epochs_array = [100, 150, 200]
    bsize_array = [8, 16, 32, 64]
    lrate_array = [0.01, 0.1, 0.3, 0.5, 1]
    stdDev_array = [0.2, 0.4, 0.6, 0.8, 1]
    
    dataframe = pd.DataFrame()
    # define different type of architecture
    for width_x in widths_array:
        for layers_x in layers_array:
            nn_architecture = {
            'layers':[(width_x, 2)],
            'activations':[relu]
            }
            for l in range(layers_x):
                nn_architecture.__getitem__('layers').append((width_x, width_x))
                nn_architecture.__getitem__('activations').append(relu)
            nn_architecture.__getitem__('layers').append((1, width_x))
            nn_architecture.__getitem__('activations').append(sigmoid)
            # for each parameter
            for epochs_x in epochs_array:
                for bsize_x in bsize_array:
                    for lrate_x in lrate_array:
                        for stdDev_x in stdDev_array:    
                            mlp = MLP(nn_architecture, 0, stdDev_x)
                            mlp.train(df_train,df_test, epochs_x, bsize_x, lrate_x)
                            predictions,y_test = mlp.predict(df_test)
                            accuracy = accuracy_score(predictions,y_test)
                            item_x = pd.DataFrame([[width_x, layers_x+1, epochs_x, bsize_x, lrate_x, stdDev_x, accuracy]],
                                                  columns=('width', 'depth layer', 'epochs', 'batch', 'lrate', 'stdDev', 'accuracy'))
                            dataframe = dataframe.append(item_x)
    
    fig = px.parallel_coordinates(dataframe, color="accuracy", 
                                  dimensions=['width', 'depth layer', 'epochs', 'batch', 'lrate', 'stdDev', 'accuracy'],
                                          color_continuous_scale=px.colors.diverging.RdYlGn,
                                          color_continuous_midpoint=0.5)
    plot(fig)
    return dataframe

def hyperparameterWorstBest(df_train, df_test):
    n_epochs = 200
    mean = 0
    bSize = 16
    worstStdDev = 0.6
    worstLRate = 0.001
    bestStdDev = 0.9
    bestLRate = 1
    nn_architecture = {
            'layers':[(10,2),(10,10),(1,10)],
            'activations':[relu,relu,sigmoid]
            }
    
    worstMlp = MLP(nn_architecture, mean, worstStdDev)
    worstMlp.train(df_train, df_test, n_epochs, bSize, worstLRate)
    worstPredictions,y_test = worstMlp.predict(df_test)
     
    plt.plot(range(worstMlp.n_epochs),worstMlp.cost)
    plt.plot(range(worstMlp.n_epochs),worstMlp.cost_validation,'r')
    plt.title('MSE error over worst model')
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.legend(['Error on Training','Error on test'])
    plt.show()
    
    bestMlp = MLP(nn_architecture, mean, bestStdDev)
    bestMlp.train(df_train, df_test, n_epochs, bSize, bestLRate)
    bestPredictions,y_test = bestMlp.predict(df_test)
     
    plt.plot(range(bestMlp.n_epochs),bestMlp.cost)
    plt.plot(range(bestMlp.n_epochs),bestMlp.cost_validation,'r')
    plt.title('MSE error over best model')
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.legend(['Error on Training','Error on test'])
    plt.show()
    
if __name__=='__main__':
    
    
    df_train = pd.read_excel(r'C:\Users\pberi\Google Drive (pberizz@gmail.com)\TUe\2IMM20 - Data mining\HW3Atrain.xlsx')
    df_test = pd.read_excel(r'C:\Users\pberi\Google Drive (pberizz@gmail.com)\TUe\2IMM20 - Data mining\HW3Avalidate.xlsx')
    
    df_test_sort = df_test.sort_values(by=['y', 'X_0'])
    #tryWithZero()
    #heatmapAccuracy()
    differentActivations(df_train, df_test_sort)
    #dataframe = hyperparamOptimization(df_train)
    #hyperparameterWorstBest(df_train, df_test)
        
    
            

    
