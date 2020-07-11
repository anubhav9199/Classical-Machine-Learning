import numpy as np
import pandas as pd

from Tree.DecisionTreeFunction import DecisionTreeAlgorithm, DecisionTreePrediction

def bootstrapping(train_df,n_bootstrap):
    """

    Parameters
    ----------
    train_df : training Data (Pandas DataFrame).
    n_bootstrap : bootstap value (Integer Value).
    
    The Function is use to make different bootstrap data for different trees.
    Bootstraped dataframe is use in different trees this bootstrap data make a
    tree to fit over a training data.
    
    Returns
    -------
    bootstrapped : Bootstraped DataFrame (Pandas DataFrame)

    """
    bootstrap_indices = np.random.randint(low = 0, high = len(train_df), size = n_bootstrap)
    bootstrapped = train_df.iloc[bootstrap_indices]
    return bootstrapped

def RandomForestAlgorithm(train_df, n_trees, mltask, n_bootstrap, n_feature, dt_max_depth):
    """
    
    train_df, 
    n_trees, 
    mltask, 
    n_bootstrap, 
    n_feature, 
    dt_max_depth

    Parameters
    ----------
    train_df : Traiining DataFrame (Pandas DataFrame).
    n_trees : No. Of tree user want to create (Integer Value).
    mltask : ML Task ("Regression" or "Classification")
    n_bootstrap : No. of Bootstrap indices create for data. (Integer Value).
    n_feature : No. of feature  (Integer Value).
    dt_max_depth : Maximum depth (Integer Value).
    
    The Function is use to fit the model of decision tree over training data. 
    the function is work over different bootstrap dataset and fit over data
    for prediction.

    Returns
    -------
    forest : list of all the trees create a forest. (List).
    
    """
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree = DecisionTreeAlgorithm(df_bootstrapped, mltask = mltask, 
                                     max_depth = dt_max_depth, 
                                     random_subspace = n_feature)
        forest.append(tree)
    
    return forest

def RandomForestPrediction(test_df, forest):
    """
    

    Parameters
    ----------
    test_df : Testing Data (Pandas DataFrame).
    forest : Forest of Trees (list of trees from Ramdon forest Algorithm).
    
    The function is use for the prediction of the fitted model. the model is
    fitted over the training data.

    Returns
    -------
    random_forest_prediction : DataFrame containing all the label predicted by
    the prediction model (Pandas DataFrame).

    """
    df_prediction = {}
    for i in range(len(forest)):
        column_name = "Tree_{}".format(i)
        prediction = DecisionTreePrediction(test_df, tree = forest[i])
        df_prediction[column_name] = prediction

    df_prediction = pd.DataFrame(df_prediction)
    random_forest_prediction = df_prediction.mode(axis=1)[0]
    return random_forest_prediction