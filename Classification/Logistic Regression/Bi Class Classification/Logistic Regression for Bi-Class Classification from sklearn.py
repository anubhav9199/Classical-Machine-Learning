#======================================================================
#Logistic Regression from Scratch over Breast Cancer Data(Bi-Class Classification)
"""
Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression). It is a Discriminate Learning Algorithm which means that it try to find posterior probability over classes directly without the envolvement of likelihood probabilities.

In statistics, the logistic model is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead, True/False or healthy/sick.

This can be extended to Classify several classes of events such as determining whether an image contains a cat, dog, lion, etc.

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
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,plot_precision_recall_curve,plot_roc_curve
#======================================================================

#======================================================================
#Prediction Function
"""
Prediction function is use to predict the accuracy of the classifier model over the train data and test data.
"""
def prediction(model,testing_data,testing_label):
    predict = model.predict(testing_data)
    accuracy = np.count_nonzero(np.equal(predict,testing_label))
    print("Accuracy : ",accuracy*100/testing_label.shape[0])
    return accuracy,predict
#======================================================================

#======================================================================
#Confusion Matrix Function
"""
A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known. The confusion matrix itself is relatively simple to understand, but the related terminology can be confusing.<br>
"""
#Precision means how much accurately our model predicts or we can say how much positive idenfied label actually correct.<br>
#Explanation of formula see in python notebook
#Recall means how much positive identified label predicted correctly.<br>
#Explanation of formula see in python notebook
def confusion_matrix_with_plot(model,testing_data,testing_label,predict_label):
    tn,fp,fn,tp = confusion_matrix(testing_label,predict_label).ravel()
    plot_confusion_matrix(model,testing_data,testing_label)
    precision = tp/(tp+fp)*100
    recall = tp/(tp+fn)*100
    return print("True Positive : ",tp,"\nFalse Positive : ",fp,"\nTrue Negative : ",tn,"\nFalse Negative : ",fn,"\nPrecision : ",precision,"\nRecall : ",recall)
#======================================================================

#======================================================================
#Main Function
def main():
    #======================================================================
    #Calling Data
    """
    Data is call for work. The Columns are selected here is according to the BREAST CANCER DATASET from WINCONSIN Hospital Easily find on Kaggle(www.kaggle.com).
    """
    data= pd.read_csv('Breast_Cancer_Data.csv')
    """
    Removing unnecessary columns from the dataset so that you won't face any trouble regarding the dataset. I use Breast Cancer Dataset to train model and predict whether the person is having cancer or not.<br>
    Preprocessing the data before applying the Logistic Classifier
    """
    data['diagnosis'].replace(to_replace=['B','M'],value=[0,1],inplace=True)
    data.drop(labels=['id','Unnamed: 32'],axis=1,inplace=True)
    #======================================================================

    #======================================================================
    #Spliting Independent and Dependent Variables
    x=data.iloc[:,1:]
    y=data.iloc[:,0]
    #======================================================================

    #======================================================================
    #Splitting training and testing data
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
    #======================================================================

    #======================================================================
    ### Normalising the data
    """
    Scaling the features.
    The scaling is also know as standardisation/normalisation. Standardization is a process for making a dataset fit for the training of the model. In this prosess we make a dataset whose value lies between zero mean and one standard deviation. The Data Comming out from this process is smooth for the curves and fitting a model.
    """
    sc=StandardScaler()
    xtrain=sc.fit_transform(xtrain)
    xtest=sc.transform(xtest)
    #=====================================================================

    #======================================================================
    #Logistic Regression Model
    """
    Creating a Logistic regression model object and fitting it over training data.<br>
    The Model contain these three function for the fitting over the data.
    """
    #--------------------------------------------------------------------------------------------------------------------------------------
    #Sigmoid Function
    """
    Defining the Sigmoid Function. Sigmoid function is use in the ml to get the probability value (i.e. between 0 to 1) for any feature. The function basically has the formula which make the value equal to probable value of the feature between 0 to 1.<br>
    """ 
    #Explanation of forumla see in python notebook
    #--------------------------------------------------------------------------------------------------------------------------------------
    #Negative Log Loss Function
    """
    Defining Function for Negative Log Loss calculation.In Convex Optimization we find the global minima to train the model so it get less error while prediction for the testing and training data. This error is called  as loss. We introduce log and a negative sign in case to smoothing of and inverting the parabola to find the global minima.
    Here, Logrithm is use to smoothing out the curve so that it don't stucked in any local minima. 
    Here, Negative sign is introduce to invert the parabola of the function.
    """
    #Explanation of formula see in python notebook
    #--------------------------------------------------------------------------------------------------------------------------------------
    #Gradient Descent
    """
    The Fit function is to fit the learning curve and reduce the loss of the model. The algorithm use here is the Stocastic Gradient Descent(SGD). The algorithm is containing the learning rate,epsilon for the stoping of under going algorithm to find global minima.
    """
    logistic=LogisticRegression(verbose=1)
    logistic.fit(xtrain,ytrain)
    #======================================================================

    #======================================================================
    #Testing our Model on the Training Data
    train_accuracy,train_predict = prediction(logistic,xtrain,ytrain)
    #Testing our Model on the Testing Data
    test_accuracy,test_predict = prediction(logistic,xtest,ytest)
    #======================================================================

    #======================================================================
    #Confusion Matrix over training data
    confusion_matrix_with_plot(logistic,xtrain,ytrain,train_predict)
    #Confusion Matrix over testing data
    confusion_matrix_with_plot(logistic,xtest,ytest,test_predict)
    #======================================================================

    #======================================================================
    #Precision-Recall Curve
    """
    The precision-recall curve shows the tradeoff between precision and recall for different threshold. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate.
    """
    #Precision-Recall Curve over training data
    plot_precision_recall_curve(logistic,xtrain,ytrain)
    #Precision-Recall Curve over testing data
    plot_precision_recall_curve(logistic,xtest,ytest)
    #======================================================================

    #======================================================================
    #Plot ROC Curve Function
    """
    Preparing to plot ROC curve. The function plot the ROC Curve from the list containing the value of true positive and false positive rate. A receiver operating characteristic curve, or ROC curve, is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. The ROC curve is created by plotting the true positive rate against the false positive rate at various threshold settings.
    """
    #Plotting ROC Curve for Training Data
    plot_roc_curve(logistic,xtrain,ytrain)
    #Plotting ROC Curve Testing Data
    plot_roc_curve(logistic,xtest,ytest)
    #======================================================================
#======================================================================

#======================================================================
#Calling Main Function
if __name__ == "__main__":

    #Calling Main Function
    main()
#======================================================================