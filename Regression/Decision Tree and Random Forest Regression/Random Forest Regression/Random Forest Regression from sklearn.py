# -----------------------------------------------------------------------------
# Random Forest Regression from Scratch using Tree Function
"""
Every decision tree has high variance, but when we combine all of them together 
in parallel then the resultant variance is low as each decision tree gets 
perfectly trained on that particular sample data and hence the output doesn’t 
depend on one decision tree but multiple decision trees. In the case of a 
classification problem, the final output is taken by using the majority voting 
classifier. In the case of a regression problem, the final output is the mean 
of all the outputs.
A Random Forest is an ensemble technique capable of performing both regression 
and classification tasks with the use of multiple decision trees and a 
technique called Bootstrap and Aggregation, commonly known as bagging. The 
basic idea behind this is to combine multiple decision trees in determining 
the final output rather than relying on individual decision trees.
Random Forest work on three functions :-
(1) Boot Strapping
(2) Random Subspace
(3) Prediction
"""
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#Ploting The Learning Curve of the model Function
"""
The Function plots the learning curve of the model over the training score and 
the cross-validation score.
Funtion to plot Learning curve from sklearn 
(https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)
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
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def main():
    # -------------------------------------------------------------------------
    # Calling Data
    """
    Data is call for work. The Columns are selected here is according to the 
    BREAST CANCER DATASET from WINCONSIN Hospital Easily find on 
    Kaggle(www.kaggle.com).
    Creating Dataset from original columns from the dataset so that you won't 
    face any trouble regarding the dataset. I use Breast Cancer Dataset to 
    train model and predict whether the person is having cancer or not.
    """
    cancer_df = pd.read_csv('./Dataset/Breast Cancer Dataset/Breast_Cancer_Data.csv')
    cancer_df['diagnosis'].replace(['B', 'M'], [0, 1], inplace =True)
    cancer_df['label'] = cancer_df.diagnosis
    cancer_df.drop([cancer_df.columns[0], cancer_df.columns[1], 
                    cancer_df.columns[32]], axis = 1, inplace = True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Preparing Data For Model
    cancer_data = cancer_df.iloc[:,0:1].values
    cancer_labels = cancer_df.iloc[:,-1].values
    cancer_data = cancer_data.reshape(cancer_data.shape[0],1)
    cancer_labels = cancer_labels.reshape(cancer_labels.shape[0],1)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Train Test Spliting
    train_cancer, test_cancer, train_labels, test_labels = train_test_split(
        cancer_data,cancer_labels,test_size = 0.25)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Fitting the Model over Data
    Forest_cancer=RandomForestRegressor()
    Forest_cancer.fit(train_cancer,train_labels)
    # -------------------------------------------------------------------------

    # Predicting the result
    # -------------------------------------------------------------------------
    # Training Data
    train_predict = Forest_cancer.predict(train_cancer)
    train_mse = mean_squared_error(train_labels,train_predict)
    train_mae = mean_absolute_error(train_labels,train_predict)
    train_score = -(round(Forest_cancer.score(train_labels,train_predict)))*100
    
    print("----------------------------------------------------------\n")
    print("Mean Square Error Training Data ; {}".format(train_mse))
    print("\nMean Absolute Error Training Data: {}".format(train_mae))
    print("\nScore Value of Training Data : {}".format(train_score))
    print("\n----------------------------------------------------------")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Testing Data
    test_predict = Forest_cancer.predict(test_cancer)
    test_mse = mean_squared_error(test_labels,test_predict)
    test_mae = mean_absolute_error(test_labels,test_predict)
    test_score = -(round(Forest_cancer.score(test_labels,test_predict)))*100
    
    print("----------------------------------------------------------\n")
    print("Mean Square Error Testing Data : {}".format(test_mse))
    print("\nMean Absolute Error Testing Data : {}".format(test_mae))
    print("\nScore Value of Testing Data : {}".format(test_score))
    print("\n----------------------------------------------------------")
    # -------------------------------------------------------------------------
    
    # Plotting the regression line
    # -------------------------------------------------------------------------
    # Training Data
    X_grid = np.arange(min(train_cancer), max(train_cancer), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(train_cancer, train_labels, color = 'red')
    plt.plot(X_grid, Forest_cancer.predict(X_grid), color = 'blue')
    plt.title('True or False (Random Forest Regression)')
    plt.ylabel('labels')
    plt.xlabel('Feature')
    plt.show()
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Testing Data
    X_grid = np.arange(min(test_cancer), max(test_cancer), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(test_cancer, test_labels, color = 'red')
    plt.plot(X_grid, Forest_cancer.predict(X_grid), color = 'blue')
    plt.title('True or False (Random Forest Regression)')
    plt.ylabel('labels')
    plt.xlabel('Feature')
    plt.show()
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    #Ploting the Learning Curve Training Data
    """
    ploting the learning curve for the Random Forest Regression which works over 
    data.
    """
    plot_learning_curve(Forest_cancer,"Learning Curve (Random Forest Train)",
                        train_cancer, train_cancer, cv=5)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    #Ploting the Learning Curve Testing Data
    """
    ploting the learning curve for the Random Forest Regression which works over 
    data.
    """
    plot_learning_curve(Forest_cancer,"Learning Curve (Random Forest Test)",
                        test_cancer, test_cancer, cv=5)
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#Calling Main Function
if __name__ == "__main__" :
    
    #Calling Main Function
    main()
# -----------------------------------------------------------------------------