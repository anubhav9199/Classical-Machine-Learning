#======================================================================
# Logistic Regression from SKLearn over MNIST Dataset (Multi-Class Classification)
"""
Logistic regression is a statistical model that in its basic form uses a logistic function to model a multiple dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary/multiple regression). It is a Discriminate Learning Algorithm which means that it try to find posterior probability over classes directly without the envolvement of likelihood probabilities.<br>

In statistics, the logistic model is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead, True/False or healthy/sick. <br>

This can be extended to Classify several classes of events such as determining whether an image contains a cat, dog, lion, etc.<br>

This code contains only about how we can fit a logistic model over user given dataset and also to get a good output result out of it. The code written keeping vision of object oriented programing which means that the code is fully moduler so that to keep in mind about the use of the functions in other programs also.
"""
#======================================================================

#======================================================================
#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
#======================================================================

#======================================================================
#Prediction Function
"""
Prediction function is use to get the accuracy of the fitted model over training and testing data so that we can get to know how much accurate over model is trained over training data to predict right output.
"""
def accuracy_with_confusion_matrix(model,testing_data,testing_labels):
    predict = model.predict(testing_data)
    accuracy = model.score(testing_data,testing_labels)
    accuracy = accuracy*100
    print('Accuracy : ',accuracy)
    cm = confusion_matrix(testing_labels,predict)
    print("\n\nThe Confusion Matrix is : \n\n",cm,"\n\nThe Graph Plot over Confusion Matrix : \n")
    plot_confusion_matrix(model,testing_data,testing_labels)
    
#======================================================================

#======================================================================
#Main Function
def main():
    #======================================================================
    #Calling Data
    """
    Data is call for fitting of object classifer created here. The Columns are selected here is according to the DIGIT RECOGNISATION DATASET from Famous MNIST DATASET Easily find on Kaggle(https://www.kaggle.com/oddrationale/mnist-in-csv?select=mnist_train.csv).
    """
    training_data = pd.read_csv('mnist_train.csv')
    testing_data = pd.read_csv('mnist_test.csv')
    #======================================================================
    
    #======================================================================
    #Preprocessing of training and testing data
    training_labels = training_data['label']
    training_data.drop(['label'],axis=1,inplace=True)
    testing_labels = testing_data['label']
    testing_data.drop(['label'],axis=1,inplace=True)
    #======================================================================
    
    #======================================================================
    #Normalising the data
    """
    Scaling the features.
    
    The scaling is also know as standardisation/normalisation. Standardization is a process for making a dataset fit for the training of the model. In this prosess we make a dataset whose value lies between zero mean and one standard deviation. The Data Comming out from this process is smooth for the curves and fitting a model.
    """
    sc=StandardScaler()
    training_data=sc.fit_transform(training_data)
    testing_data=sc.fit_transform(testing_data)
    #======================================================================
    
    #======================================================================
    #Logistic Regression Model
    """
    Creating a Logistic regression model object and fitting it over training data.<br>
    The Model contain these three function for the fitting over the data."""
    #--------------------------------------------------------------------------------------------------------------------------------------
    #Softmax Function
    """
    Defining the Softmax Function. Softmax function is use in the ml to get the probability value (i.e. between 0 to 1) for any feature. The function basically has the formula which make the value equal to probable value of the feature between 0 to 1. Funtion to return Posterior Probabilities
    """
    #Formula : refer to python notebook
    #--------------------------------------------------------------------------------------------------------------------------------------
    #Derivative Functions
    """
    Derivative functions are define to find the derivative of the features to train the model and get the weights for the GDA. Funtions to return derivatives with respect to wieghts for GDA.
    """
    #Formula del by del0s : refer to python notebook
    #Formula del by dels : refer to python notebook
    #--------------------------------------------------------------------------------------------------------------------------------------
    #Cross Entropy Function 
    """
    Defining Function for Cross Entroy Loss calculation.In Convex Optimization we find the global minima to train the model so it get less error while prediction for the testing and training data. This error is called  as loss. We introduce log and a negative sign in case to smoothing of and inverting the parabola to find the global minima.<br>
    """
    #Here, Logrithm is use to smoothing out the curve so that it don't stucked in any local minima. 
    #Here, Negative sign is introduce to invert the parabola of the function.
    #Formula : refer to python notebook
    #--------------------------------------------------------------------------------------------------------------------------------------
    #Gradient Descent
    """
    The Fit function is to fit the learning curve and reduce the loss of the model. The algorithm use here is the Stocastic Gradient Descent(SGD). The algorithm is containing the learning rate,epsilon for the stoping of under going algorithm to find global minima.Creating Classifier object and fitting to our training data
    """
    logistic=LogisticRegression(verbose=1,multi_class='ovr',max_iter=10000)
    logistic.fit(training_data,training_labels)
    #======================================================================
    
    #======================================================================
    #Predicting result on both the datasets
    #Training Accuracy
    accuracy_with_confusion_matrix(logistic,training_data,training_labels)
    #Testing Accuracy
    accuracy_with_confusion_matrix(logistic,testing_data,testing_labels)
    #======================================================================
#======================================================================

#======================================================================
#Calling Main Function
if __name__ == "__main__":
    
    #Calling Main Function
    main()
#======================================================================