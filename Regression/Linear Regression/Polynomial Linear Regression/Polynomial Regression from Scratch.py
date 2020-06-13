#======================================================================
#Polynomial Linear Regression from Scratch on Breast Cancer Data
"""Linear regression is a algorithm which is starting level algorithm for machine learning prediction. This code contains 
only about how we can fit a linear model over user given dataset and also to get a good output result out of it. The code 
is fully moduler so that to keep in mind about the use of the functions in other programs also."""
#======================================================================

#======================================================================
#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#======================================================================

#======================================================================
#Calling Data
"""Data is call for work. The Columns are selected here is according to the BREAST CANCER DATASET from WINCONSIN Hospital 
Easily find on Kaggle(www.kaggle.com)."""
#read the data
data = pd.read_csv('Breast_Cancer_Data.csv')
#Preprocessing the data according to the given dataset.
"""Removing unnecessary columns from the dataset so that you won't face any trouble regarding the dataset. I use Breast 
Cancer Dataset to train model and predict whether the person is having cancer or not."""
data.drop([data.columns[0],data.columns[32]],axis=1,inplace=True)
data['diagnosis'].replace(to_replace=['B','M'],value=[0,1],inplace=True)
#======================================================================

#======================================================================
"""Polynomial Regression is done by introducing higher degree terms in our regression problem to make it easier to fit 
a line through the data Here x only has one feature, therefore we will introduce higher degree terms in it 
for example :- $ x^2,x^3,etc $"""
#======================================================================

#======================================================================
#Objective Function
"""The objective function is the Mean Square Function for the Polynomial Regression. Mean square error function shows 
the mean square error done by the gradient descent to lower down the error/loss in the learning and predicting of 
data."""
def objective_function(x,y,theta,theta_0,lam=0):
    MSE = np.sum(np.square(x@theta + theta_0 - y) + lam*0.5*np.square(np.linalg.norm(theta)))
    return (MSE/x.shape[0]*2)
#======================================================================

#======================================================================
#Theta Function
"""The fdash_theta and fdash_theta0 function are the function to calculate the theta and theta0 for the gradient descent 
for the training of the linear model."""
def fdash_theta(x,y,theta,theta_0,lam=0):
    ans = (x@theta+theta_0-y ).T@x + lam*np.linalg.norm(theta)
    return 2*ans.T/x.shape[0]

def fdash_theta0(x,y,theta,theta_0):
    ans = np.sum(x@theta+theta_0-y)
    return 2*ans/x.shape[0]
#======================================================================

#======================================================================
#Data Creation Function
"""The data creation function is use to create an augmented data of the original data. The augmented data is containing 
the value of the user provide data having degree 2, 3, or more. for example :- $ x^2,x^3,etc $"""
def data_creation(x,y,degree):
    dummy1 = x
    dummy_temp = x
    
    for d in range(degree-1):
        dummy_temp = x*dummy_temp
        dummy1 = np.append(dummy1,dummy_temp,axis=1)
        
    # new data with the higher degree terms
    augmented_data = dummy1
    
    # normalise data
    xmean = np.mean(augmented_data,axis=0)
    xstd = np.std(augmented_data,axis=0)
    augmented_data = (augmented_data-xmean)/xstd
    
    ymean = np.mean(y)
    ystd = np.std(y)
    y = (y-ymean)/ystd
    
    del dummy1,dummy_temp
    
    return augmented_data,y
#======================================================================

#======================================================================
#Plotter Function
"""The plotter function is use to plot the graph of polynomial regression. The graph is plot over different degree of 
the data."""
def plotter(x,y,degree,theta):
    dummy1 = x
    dummy_temp = x

    for d in range(degree-1):
        dummy_temp = x*dummy_temp
        dummy1 = np.append(dummy1,dummy_temp,axis=1)

    # new data with the higher degree terms
    augmented_data = dummy1

    ymean = np.mean(y)
    ystd = np.std(y)
    xmean = np.mean(augmented_data,axis=0)
    xstd = np.std(augmented_data,axis=0)
    plt.plot(x,(((augmented_data-xmean)/xstd@theta)*ystd+ymean),label='degree '+str(degree))
#======================================================================

#======================================================================
#Polynomial Regression Function
"""The Fit function is to fit the learning curve and reduce the loss of the model. The algorithm use here is the 
Stocastic Gradient Descent(SGD). The algorithm is containing the learning rate,epsilon for the stoping of under 
going algorithm to find global minima."""
def polynomial_regression(x,y,degree,alpha=0.001,eps=10e-10,lam=0):
    """
    This function will fit the given data x to y
    
    x: array containing the data
    y: array containing regression values
    degree: int value stating the highest degree polynomial to be introduced
    alpha=0.001: Learning Rate
    eps=10e-4: tolerance 
    lam=0: value of L2 regularisation parameter
    """     
    
    augmented_data,y = data_creation(x,y,degree)
    # Gradient Descent
    
    theta_i = np.zeros(shape=(augmented_data.shape[1],1))
    theta_0i = np.zeros(1)
    
    i = 0
    
    #Loop for the continuing the process again till it gets to the local minima
    while True:
        
        mse_i = objective_function(augmented_data,y,theta_i,theta_0i,lam)
        
        theta_f = theta_i - alpha*fdash_theta(augmented_data,y,theta_i,theta_0i,lam)
        theta_0f = theta_0i - alpha*fdash_theta0(augmented_data,y,theta_i,theta_0i)
        
        mse_f = objective_function(augmented_data,y,theta_f,theta_0f,lam)
        
        if i%50000==0:
            print("MSE at step {} is {}".format(i,mse_f))
        i+=1
        
        if abs(mse_f-mse_i)<=eps:
            print("MSE at step {} is {}".format(i,mse_f))
            break
                
        theta_i = theta_f
        theta_0i = theta_0f
        
    return theta_f,theta_0f
#======================================================================
    
#======================================================================
#Simple Linear Regression
"""In case of Linear Regression. The linear model gives theta from the given formula. :- """
#𝜃=(𝑋𝑇𝑋)−1𝑋𝑇𝑌
"""This formula is calculating the weight from simple linear regression model. Simple Linear regression is a 
algorithm which gives an overview of the theta over single feature of the dataset. """

"""Simple Linear Regression function is a function use to calculate theta(weight) over a particular feature 
given by the user. Here a concept of polynomial regression of creating augmented data is used and the theta 
is calculated over the augmented data."""
def simple_lr(x,y,degree):
    dummy1 = x
    dummy_temp = x

    for d in range(degree-1):
        dummy_temp = x*dummy_temp
        dummy1 = np.append(dummy1,dummy_temp,axis=1)

    # new data with the higher degree terms
    augmented_data = dummy1

    ymean = np.mean(y)
    ystd = np.std(y)
    xmean = np.mean(augmented_data,axis=0)
    xstd = np.std(augmented_data,axis=0)

    adj_data = (augmented_data-xmean)/xstd
    adj_labels = (y-ymean)/ystd

    theta = np.linalg.inv(adj_data.T @ adj_data)@adj_data.T@adj_labels
    
    return theta
#======================================================================

#======================================================================
#Main Function
def main():
    
    #======================================================================
    #Preparing of data
    """The data is divided in x and y terms by taking label as y and any of the feature as x.here i use perimeter mean in 
    the training of linear model."""
    x = data.iloc[:,1:2]
    y = data.iloc[:,0:1]
    #======================================================================
    
    #======================================================================
    # since they are series
    x = np.array(x).reshape(569,1)
    y = np.array(y).reshape(569,1)
    theta,theta_0 = polynomial_regression(x,y,5,lam=0.05,alpha=10e-4,eps=10e-10)
    #======================================================================
    
    #======================================================================
    #Calculating Theta
    """Calculating the theta over various degree of the given data. The data is given by the user according the 
    regression user wants. The degree is of the data which is :- x^2, x^3, etc"""
    for d in range(1,4):
        theta,theta_0 = polynomial_regression(x,y,d,lam=0.1,alpha=10e-4,eps=10e-10)
        plotter(x,y,d,theta)
    plt.scatter(x,y,label='original',color='red')
    plt.legend()
    #======================================================================
    thet = simple_lr(x,y,degree = 5)
    print("The Theta Calculate Simple : ",thet)
    #======================================================================
#======================================================================

#======================================================================
#Calling Main Function
if __name__ == "__main__":
    
    #MAIN FUNCTION
    main()