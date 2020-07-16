#======================================================================
#Decision tree regression from SKLearn over Breast Cancer Dataset
"""
Decision tree builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. A decision node (e.g., Cancer) has two or more branches (e.g., Benign and Malignant), each representing values for the attribute tested. Leaf node (e.g., Radius Mean,Texture Mean) represents a decision on the numerical target. The topmost decision node in a tree which corresponds to the best predictor called root node. Decision trees can handle both categorical and numerical data.
"""
#======================================================================

#======================================================================
#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import learning_curve
#======================================================================

#======================================================================
#Mean Square & Mean Absolute Error
"""
Function returns the value error in the training of the model while predicting the target variable.<br>
"""
#Mean Square Error :
"""
Mean Square Error (MSE). It is the standard deviation of how far from the regression line data points are. In other words, it tells you how concentrated the data is around the line of best fit.
"""
#Mean Absolute Error :
"""
Mean Absolute Error (MAE). It is the absolute standard deviation of how far from the regression line data points are.
"""
#Score :
"""
R^2 Score function is use to return the score value in the data provide.
"""
def calculate_error(model,test_data,test_label):
    predict = model.predict(test_data)
    mse = mean_squared_error(test_label,predict)
    mae = mean_absolute_error(test_label,predict)
    score = model.score(test_data,test_label)
    return mse,mae,score
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
#Main Function
def main():
    #======================================================================
    #Calling Data
    """
    Data is call for work. The Columns are selected here is according to the BREAST CANCER DATASET from WINCONSIN Hospital Easily find on Kaggle(www.kaggle.com).
    """
    data = pd.read_csv('./Dataset/Breast Cancer Dataset/Breast_Cancer_Data.csv')
    """
    Removing unnecessary columns from the dataset so that you won't face any trouble regarding the dataset. I use Breast Cancer Dataset to train model and predict whether the person is having cancer or not.
    """
    data.drop([data.columns[0],data.columns[32]],axis=1,inplace=True)
    data['diagnosis'].replace(["B","M"],[0,1],inplace=True)
    #======================================================================
    
    #======================================================================
    #Preprocessing Data
    x = data.iloc[:,3:4].values
    y = data['diagnosis'].values
    #======================================================================

    #======================================================================
    #Regression Fitting
    """
    Fitting the regressor to the data
    """
    regressor=DecisionTreeRegressor()
    regressor.fit(x,y)
    #======================================================================
    
    #======================================================================
    #Predicting the result
    mse,mae,score = calculate_error(regressor,x,y)
    print("Mean Square Error in Data : {}\nMean Absolute Error in Data : {}\nScore in Data : {}".format(mse,mae,score))
    #======================================================================
    
    #======================================================================
    #Plotting the regression line
    plt.scatter(x,y,color='red')
    plt.plot(x,regressor.predict(x),color='blue')
    plt.title('Decision Tree Regression')
    plt.xlabel('Feature Value')
    plt.ylabel('Cancer')
    plt.show()
    #======================================================================
    
    #======================================================================
    #Ploting the Learning Curve
    """
    ploting the learning curve for the polynomial linear model which works over data of degree 3.
    """
    plot_learning_curve(regressor,"Learning Curve (Decision Tree)",x,y,cv=5)
    #======================================================================
#======================================================================

#======================================================================
#Calling Main Function
if __name__ == "__main__" :
    
    #Calling Main Function
    main()
#======================================================================