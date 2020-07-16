#======================================================================
#Multiple Linear Regression From Scratch on Brest Cancer Data
"""Linear regression is a algorithm which is starting level algorithm for machine learning prediction. This code contains only about 
how we can fit linear model over any data and get a good output result out of it. The code is fully library base so that to keep in 
mind about the use of the functions of library in other programs also."""
#======================================================================

#======================================================================
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.linear_model import LinearRegression
#======================================================================

#======================================================================
#Funtion to plot Learning curve from sklearn (https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)
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
#MAIN FUNCTION
def main():
    
    #======================================================================
    #Data Calling
    data=pd.read_csv('./Dataset/Breast Cancer Dataset/Breast_Cancer_Data.csv')
    #======================================================================

    #======================================================================
    #Data Processing according to the dataset use.
    data.drop([data.columns[0],data.columns[32]],axis=1,inplace=True)
    data['diagnosis'].replace(to_replace=['B','M'],value=[0,1],inplace=True)
    #======================================================================
    
    #======================================================================
    #Splitting dependent and independent variables
    y=data.iloc[:,0:1].values
    x=data.iloc[:,1:5].values
    #======================================================================
    
    #======================================================================
    #Spliting of data in training and testing part for the train and testing of the model.
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    #======================================================================
    
    #======================================================================
    #Creating Linear Model
    """Linear Model created here is by using the linearmodel function of the SKLEARN Library. The model is actual 
    Linear Regression model which is use data for fittin/training."""
    linreg=LinearRegression()
    #======================================================================
    
    #======================================================================
    #Fitting of Model
    """The model created is fitted over the training data to get a result/prediction."""
    linreg.fit(x_train,y_train)
    #======================================================================
    
    #======================================================================
    #Predicting the output and plotting predicted output with actual output (which should be close to x=y line)
    y_pred=linreg.predict(x_test)
    #======================================================================
    
    #======================================================================
    #Ploting the graph
    """Ploting the curve over the train weights from the model and the testing data to see or check our prediction."""
    plt.scatter(y_pred,y_test,color='red')
    plt.plot(linreg.predict(x_train),y_train,color='blue')
    #======================================================================
    
    #======================================================================
    #Ploting Learning Curve
    plot_learning_curve(linreg,'Learning Curve',x_train,y_train,cv=5)
    #======================================================================
#======================================================================
    
#======================================================================
#Calling Main Function
if __name__ == "__main__" :
    
    #Calling Main Function
    main()
#======================================================================