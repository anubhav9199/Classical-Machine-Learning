#======================================================================
# Polynomial Linear Regression from SKLearn on Breast Cancer Data
"""
Linear regression is a algorithm which is starting level algorithm for machine learning prediction. This code contains only about 
how we can fit a linear model over user given dataset and also to get a good output result out of it. The code is fully library base so 
that to keep in mind about the use of the functions of library in other programs also. It also demponstrate all the graph that how a 
polynomial regression works. The Differnce between Simple, Multiple and Polynomial Linear Regression is: :-

Simple LR works over single Feature with single degree only.
Multiple LR works over multiple Feature with Single degree only.
Polynomial LR works over single Feature with Polynomial degree.
"""
#for eg => x^2, x^3, etc
#======================================================================

#======================================================================
#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split,learning_curve
#======================================================================

#======================================================================
#Calling Data
"""
Data is call for work. The Columns are selected here is according to the BREAST CANCER DATASET from WINCONSIN Hospital Easily find 
on Kaggle(www.kaggle.com).
"""
data=pd.read_csv('Breast_Cancer_Data.csv')

"""
Removing unnecessary columns from the dataset so that you won't face any trouble regarding the dataset. I use Breast Cancer 
Dataset to train model and predict whether the person is having cancer or not.
"""
data.drop([data.columns[0],data.columns[32]],axis=1,inplace=True)
data['diagnosis'].replace(to_replace=['B','M'],value=[0,1],inplace=True)
#======================================================================

#======================================================================
#Polynomial Data Creation and Train Test Split Function
"""
Polynomial Data Creation and Train Test Split Function is made to create polynomial data of a selected feature of the dataset. 
The function is also split the dataset into train and test data for fitting of the model.
"""
#If you want to create single polynomial you can use the below code or use the function.
#polyreg = PolynomialFeatures(degree=2)<br>
#xpoly = polyreg.fit_transform(x)<br>
def polynomial_data_creation_and_train_test_split(x,y,degree=1,test_size=0.3):
    polyreg=PolynomialFeatures(degree=degree)
    xpoly=polyreg.fit_transform(x)
    x_train,x_test,y_train,y_test=train_test_split(xpoly,y,test_size=test_size)
    return xpoly,x_train,x_test,y_train,y_test
#======================================================================


#======================================================================
#Ploting The Learning Curve of the model Function
"""
The Function plots the learning curve of the model over the training score and the cross-validation score.
Funtion to plot Learning curve from sklearn (https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)
"""
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
#======================================================================

#======================================================================
#Ploting the graph of Linear Regression
"""
Ploting the curve over the train weights from the model and the testing data to see or check our prediction. The data above the line 
represents Cancer to be Malignent(i.e. 1 or 'M') and  the data below the line represents Benigen(i.e. 0 or 'B').
"""
def plotter(x_data,y_data,x_test,x_poly,color,linreg,label):
    plt.scatter(x_data,y_data,color='red')
    plt.plot(x_test,linreg.predict(x_poly),color=color,label=label)
    plt.legend(loc='best')
#======================================================================

#======================================================================
#MAIN FUNCTION
def main():
    
    #======================================================================
    #Preparing of data
    """
    The data is divided in x and y terms by taking label as y and any of the feature as x.here i use perimeter mean in the training 
    of linear model
    """
    x=data.iloc[:,1:2]
    y=data.iloc[:,0:1]
    #======================================================================

    #======================================================================
    #Preparing of Polynomial data (Degree = 1)
    """
    The data is divided in x and y terms by taking label as y and any of the feature as x.here i use perimeter mean in the training 
    of linear model.
    """
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
    #======================================================================
    
    #======================================================================
    #Preparing of Polynomial data (Degree = 2)
    """
    The data is divided in x and y terms by taking label as y and any of the feature as x.here i use perimeter mean in the training 
    of linear model.here previously we have single degree data but now in this segment we increase degree to 2
    """
    xpoly,xpoly_train,xpoly_test,ypoly_train,ypoly_test = polynomial_data_creation_and_train_test_split(x,y,degree=2,test_size=0.3)
    #======================================================================
    
    #======================================================================
    ### Preparing of Polynomial data (Degree = 3)
    """
    The data is divided in x and y terms by taking label as y and any of the feature as x.here i use perimeter mean in the training 
    of linear model.here previously we have single degree data but now in this segment we increase degree to 3
    """
    xpoly2,xpoly2_train,xpoly2_test,ypoly2_train,ypoly2_test = polynomial_data_creation_and_train_test_split(x,y,degree=3,test_size=0.3)
    #======================================================================
    
    #======================================================================
    #Creating Linear Model and Fitting of Model
    """
    Linear Model created here is by using the linearmodel function of the SKLEARN Library. The model is actual Linear Regression model 
    which is use data for fittin/training.
    The model created is fitted over the training data to get a result/prediction.
    """
    #Simle Linear Regression or Polynomial Linear Regression with degree 1
    linreg=LinearRegression()
    linreg.fit(x_train,y_train)
    
    #Polynomial Linear Regression with degree 2
    linreg2=LinearRegression()
    linreg2.fit(xpoly_train,ypoly_train)
    
    #Polynomial Linear Regression with degree 3
    linreg3=LinearRegression()
    linreg3.fit(xpoly2_train,ypoly2_train)
    #======================================================================
    
    #======================================================================
    #Ploting the graph of Linear Regression
    """
    Ploting the curve over the train weights from the model and the testing data to see or check our prediction. The data above the line 
    represents Cancer to be Malignent(i.e. 1 or 'M') and  the data below the line represents Benigen(i.e. 0 or 'B').
    """
    plotter(x,y,x_test=x_test,x_poly=x_test,color='blue',linreg=linreg,label="Linear Regression")
    #======================================================================
    
    #======================================================================
    #Ploting the graph of Polynomial Linear Regression (degree = 2)
    """
    Ploting the curve over the train weights from the model and the testing data to see or check our prediction. The data above the line 
    represents Cancer to be Malignent(i.e. 1 or 'M') and  the data below the line represents Benigen(i.e. 0 or 'B').
    """
    plotter(x,y,x_test=x,x_poly=xpoly,color='cyan',linreg=linreg2,label="Polynomial Regression (Degree = 2)")
    #======================================================================
    
    #======================================================================
    #Ploting the graph of Polynomial Linear Regression (degree = 3)
    """
    Ploting the curve over the train weights from the model and the testing data to see or check our prediction. The data above the line 
    represents Cancer to be Malignent(i.e. 1 or 'M') and  the data below the line represents Benigen(i.e. 0 or 'B').
    """
    plotter(x,y,x_test=x,x_poly=xpoly2,color='green',linreg=linreg3,label="Polynomial Regression (Degree = 3)")
    #======================================================================
    
    #======================================================================
    #Prediction of the Models over degree 1,2 and 3 Datasets
    """
    Predicting the output and plotting predicted output with actual output (which should be close to x=y line)
    """
    print('Prediction from:\nLinear Model:',linreg.predict(x_test),'\nPolynomial Model (Degree=2):',linreg2.predict(xpoly_test),
          '\nPolynomial Model (Degree=3):',linreg3.predict(xpoly2_test))
    #======================================================================

    #======================================================================
    #Ploting the Learning Curve SLR
    """
    ploting the learning curve for the simple linear model which works over data of degree 1.
    """
    plot_learning_curve(linreg,"Learning Curve(Linear Regression)",x_test,y_test,cv=5)
    #======================================================================
    
    #======================================================================
    #Ploting the Learning Curve PLR (Degree = 2)
    """
    ploting the learning curve for the polynomial linear model which works over data of degree 2.
    """
    plot_learning_curve(linreg2,"Learning Curve(Polynomial Regression(degree = 2))",xpoly_test,ypoly_test,cv=5)
    #======================================================================
    
    #======================================================================
    #Ploting the Learning Curve PLR (Degree = 3)
    """
    ploting the learning curve for the polynomial linear model which works over data of degree 3.
    """
    plot_learning_curve(linreg3,"Learning Curve(Polynomial Regression(degree = 3))",xpoly2_test,ypoly2_test,cv=5)
    #======================================================================
#======================================================================

#======================================================================
#Calling Main Function
if __name__=="__main__":
    
    #Calling Main Function
    main()
#======================================================================