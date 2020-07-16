# Importing Library
import random
# import pandas as pd
# import matplotlib.pyplot as plt
# from DecisionTreeFunction import predict_example

def train_test_split(df,test_size,cross_val_size=0):
    
    """
    
    df, test_size, cross_val_size
    
    PARAMETER
    ---------
    df : Pandas DataFrame
    test_size : test_size (Integer or Float) value
    cross_val_size : cross_validation_size (Integer or Float) value. (Default = 0)
    
    This Function is  use to divide the data into train and test DataFrame. If
    you give size of cross validation data it also divide data into three part
    training dataframe, testing dataframe, cross validation dataframe.
    
    Returns
    -------
    train_df : Training DataFrame. (Pandas DataFrame)
    test_df : Testing DataFrame. (Pandas DataFrame)
    cross_val_df : Cross Validation DataFrame. (Pandas DataFrame)
    
    """
    if isinstance(test_size,float):
        test_size = round(test_size * len(df))
    
    if cross_val_size != 0 :
        if isinstance(cross_val_size,float):
            cross_val_size = round(cross_val_size * len(df))
        index = df.index.tolist()
        cross_val_index = random.sample(population=index,k=cross_val_size)
        test_index = random.sample(population=index,k=test_size)
        cross_val_df = df.loc[cross_val_index]
        test_df = df.loc[test_index]
        train_df = df.drop([test_index,cross_val_df])    
        return train_df,cross_val_df,test_df
    
    else :
        index = df.index.tolist()
        test_index = random.sample(population=index,k=test_size)
        test_df = df.loc[test_index]
        train_df = df.drop(test_index)    
        return train_df,test_df

def determine_type_of_feature(df):
    
    """
    
    df
    PARAMETER
    ---------
    df : Pandas DataFrame
    
    This function is use to determine the type of feature present in the data.
    Either the data has Continuous Feature value Or the data has Catagorical 
    Feature.
    
    Returns
    -------
    Feature_types : list of all the feature telling either continuous or 
    catagorical (list)
    
    """
    
    feature_types = []
    n_unique_value_threshold = 15
    
    for column in df.columns:
        unique_values = df[column].unique()
        example_values = unique_values[0]
        
        if (isinstance(example_values, str)) or (len(unique_values) <= n_unique_value_threshold):
            feature_types.append("catagorical")
        else:
            feature_types.append("continuous")
    return feature_types

def calculate_accuracy(prediction,labels):
    """
    Returns accuracy of prediction of model.

    Parameters
    ----------
    prediction : Predicted Labels from model.
    labels : True Labels from the dataframe.

    Returns
    -------
    accuracy : Float value of accuracy.

    """
    predictions_correct = prediction == labels
    accuracy = predictions_correct.mean()
    
    return accuracy

def confusionMatrix(y_actual, y_predicted):
    """
    Returns list of confusion matrix, accuracy, precision, recall, fm

    Parameters
    ----------
    y_actual : True Value from Dataframe
    y_predicted : Predicted Value from model.

    Returns
    -------
    cm : List of  Confusion Matrix.
    accuracy : Float value of accuracy.
    precision : Float Value.
    recall : Float Value.
    fmsens : Float Value.

    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    for i in range(len(y_actual)):
        if y_actual[i] >= 0:
            if y_actual[i] == y_predicted[i]:
                tp = tp + 1
            else:
                fn = fn + 1
        if y_actual[i] <= 1:
            if y_actual[i] == y_predicted[i]:
                tn = tn + 1
            else:
                fp = fp + 1
                
    cm = [tn, fp, fn, tp]
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    fm = (2*precision*recall)/(precision+recall)
    return cm, accuracy, precision, recall, fm