#======================================================================
#Linear Regression from Sci-Kit Learn Library
"""Linear regression is a algorithm which is starting level algorithm for machine learning prediction. 
This code contains only about how we can genrate our own linear data with some noise and also to get 
a good output result out of it. The code is fully written using Sci-Kit Learn Library"""
#======================================================================

#======================================================================
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse,r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
#======================================================================

#======================================================================
#Data Genration Function
"""The function is use for the data generation it can produce linear data using linspace function of 
numpy library"""
def data_gen(n):
    x=np.linspace(0,1,n)
    y=15*x+5+np.random.randn(len(x))
    x=x.reshape(x.shape[0],1)
    y=y.reshape(y.shape[0],1)
    plt.scatter(x,y)
    plt.title('Raw Data')
    plt.show()
    return x,y
#======================================================================

#======================================================================
#Learning Curve Plotting Function
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
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
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
#MAIN FUNCTION OF THE LINEAR REGRESSION
x,y = data_gen(n=1500)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
#======================================================================

#======================================================================
#Making Linear Regression Model
lin_reg = linear_model.LinearRegression()
#======================================================================

#======================================================================
#Fitting the Model
lin_reg.fit(x_train,y_train)
#======================================================================

#======================================================================
#Training Fit
plt.scatter(x_train,y_train)
plt.plot(x_train,lin_reg.predict(x_train),color='red',linewidth=3)
plt.title('Training fit')
plt.show()
#======================================================================

#======================================================================
#Predition of testing data
y_predict = lin_reg.predict(x_test)
#======================================================================

#======================================================================
#Testing Fit
plt.scatter(x_test,y_test)
plt.plot(x_test,y_predict,color='red',linewidth=3)
plt.title('Testing fit')
plt.show()
#======================================================================

#======================================================================
#Printing all the Coefficients, Mean Square Error etc
print("Coefficients : \n",lin_reg.coef_)
print("Mean Square Error : %.2f" %mse(y_test,y_predict))
print("Coefficient of determine : %.2f" %r2_score(y_test,y_predict))
#======================================================================

#======================================================================
#Plotting all the data combine (X-test, Y-test, Prediction)
plt.grid()
plt.scatter(x_test,y_test,color = 'black')
plt.plot(x_test,y_predict,color='red',linewidth = 3)
plt.xlabel('X-Data')
plt.ylabel('Y-Data')
plt.title('Combine Data of X-test, Y-test, Prediction')
plt.show()
#======================================================================

#======================================================================
#Plotting Learning Curve using Learning curve function written above
plot_learning_curve(lin_reg,'Learning Curve',x_train,y_train,cv=5)
#======================================================================