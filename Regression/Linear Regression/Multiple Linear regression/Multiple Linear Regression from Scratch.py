#======================================================================
#Multiple Linear Regression From Scratch on Brest Cancer Data
"""Linear regression is a algorithm which is starting level algorithm for machine learning prediction. 
This code contains only about how we can genrate our own linear data with some noise and also to get 
a good output result out of it. The code is fully moduler so that to keep in mind about the use of 
the functions in other programs also."""
#======================================================================

#======================================================================
#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#======================================================================

#======================================================================
#data calling and processing
data=pd.read_csv('Breast_Cancer_Data.csv')
data.drop([data.columns[0],data.columns[32]],axis=1,inplace=True)
data['diagnosis'].replace(to_replace=['B','M'],value=[0,1],inplace=True)
#======================================================================

#======================================================================
#Standardization
"""Standardization is a process for making a dataset fit for the training of the model. In this prosess 
we make a dataset whose value lies between zero mean and one standard deviation. The Data Comming out 
from this process is smooth for the curves and fitting a model."""
def standardize(data):
    return (data-np.mean(data,axis=0))/np.std(data,axis=0)
#======================================================================
print(data.head())

#======================================================================
#Data Processing
"""The function use to process data accirding to the user for the fitting of Linear Regression model. 
The Columns are selected here is according to the BREAST CANCER DATASET from WINCONSIN Hospital Easily 
find on Kaggle(www.kaggle.com)."""
def data_process(data):
    y=data.iloc[:,0:1].values
    x=data.iloc[:,1:5]
    x=pd.get_dummies(x)
    x=np.array(x)
    y=y.reshape(y.shape[0],1)
    x=standardize(x)
    y=standardize(y)
    x=np.append(arr=np.ones((x.shape[0],1)),values=x.astype(float),axis=1)
    return x,y
#======================================================================
    
#======================================================================
#Predition Function
"""prediction function for predicting the result of the linear regression"""
def predict(theta,xi):
    
    #function returns the prediction value(RESULT)
    return xi@theta.T
#======================================================================
    
#======================================================================
#Mean Square Error
"""mean square error function shows the mean square error done by the gradient descent to lower down 
the error/loss in the learning and predicting of data"""
def sq_loss(theta,x,y):
    return (np.sum(np.power(((x@theta.T)-y),2)))/2*len(x)
#======================================================================
    
#======================================================================
#Gradient Function
"""gradient function is use to find the derivatives for the gradient descent algorithm"""
def gradients(theta,x,y):
    
    #derivaties of the gradient
    #function returns the list of gradient derivaties
    return (np.sum(x*(x@theta.T-y),axis=0))/len(x)
#======================================================================
    
#======================================================================
#Gradient Descent
"""The Fit function is to fit the learning curve and reduce the loss of the model. The algorithm use 
here is the Stocastic Gradient Descent(SGD). The algorithm is containing the learning rate,epsilon for 
the stoping of under going algorithm to find global minima"""
def fit(x,y,alpha,epsilon):
    
    #Initializing variables
    ti=np.zeros((1,x.shape[1]))
    i=0
    error=1
    loss=sq_loss(ti,x,y)
    losslist=[]
    iterlist=[]
    
    #Loop for the continuing the process again till it gets to the local minima
    while error>epsilon:
        
        #using of gradient function
        gt=gradients(ti,x,y)
        
        theta=ti-alpha*gt
        ti=theta
        
        loss_final=sq_loss(theta,x,y)
        error=abs(loss_final-loss)
        
        losslist.append(loss)
        iterlist.append(i)
        loss=loss_final
        i+=1
        
    #ploting the learning curve
    plt.plot(iterlist,losslist)
    plt.xlabel('Iterations')
    plt.title('Loss curve')
    plt.ylabel('Loss')
    plt.show()
    return theta
#======================================================================

#======================================================================
#Ploting line
"""line function to trace the line curve in the plot and to get the error/loss line 
which is y = m * x + c"""
def line(slope, intercept):
    
    axes=plt.gca()
    x=np.array(axes.get_xlim())
    
    #Line equation which is to be plot here intercept is the distance from the origin to the line on the y-axia
    #and slope is the angle of the line from the positive x-axis
    y=intercept+slope*x
    
    #ploting the line
    plt.plot(x, y,'--r',linewidth=2)
    plt.show()
#======================================================================
    
#======================================================================
#MAIN FUNCTION OF THE LINEAR REGRESSION
def main():
    
    #======================================================================
    #Preprocessing
    x,y = data_process(data)
    
    train_len=int(x.shape[0]*0.7)
    y.reshape(y.shape[0],1)
    x_train=x[:train_len,:]
    x_test=x[train_len:,:]
    y_train=y[:train_len,:]
    y_test=y[train_len:]
    #======================================================================
    
    #======================================================================
    #Fitting the Curve
    alpha=0.001
    epsilon=0.0001
    t=fit(x_train,y_train,alpha,epsilon)
    #======================================================================
    
    #======================================================================
    #Training Result
    result_train = predict(t,x_train)
    #======================================================================
    
    #======================================================================
    #Training error work
    training_error=abs(y_train-result_train)
    mean_training_error=np.sum(training_error)/training_error.shape[0]
    plt.plot(x_train,result_train,'r')
    plt.show()
    #======================================================================
    
    #======================================================================
    #Testing Result
    result_test=predict(t,x_test)
    #======================================================================
    
    #======================================================================
    #Testing error work
    testing_error=result_test-y_test
    testing_error=abs(testing_error)
    mean_testing_error=np.sum(testing_error)/testing_error.shape[0]
    plt.plot(x_test,result_test,'r')
    plt.show()
    #======================================================================
    
    #======================================================================
    #Printing all the Details of  the "Train Model"
    print("Square Loss(Training) : ",sq_loss(t,x_train,y_train))
    print("Mean Training Error   : ",mean_training_error)
    print("Square Loss(Testing)  : ",sq_loss(t,x_test,y_test))
    print("Mean Testing Error    : ",mean_testing_error)
    #======================================================================
#======================================================================

#======================================================================
#Calling Main Function
if __name__ == "__main__" :
    
    #MAIN Function acting as library
    main()