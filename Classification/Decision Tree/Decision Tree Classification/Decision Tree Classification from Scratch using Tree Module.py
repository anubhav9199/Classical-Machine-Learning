# -----------------------------------------------------------------------------
# Decision Tree Classification from Scratch using Tree Module over Two Dataset (Continuous and Catagorical) 
"""
Decision Tree algorithms are used for both predictions as well as classification 
in machine learning. Using the decision tree with a given set of inputs, one 
can map the various outcomes that are a result of the consequences or decisions.
Decision Tree Analysis is a general, predictive modelling tool that has 
applications spanning a number of different areas. In general, decision trees 
are constructed via an algorithmic approach that identifies ways to split a 
data set based on different conditions. It is one of the most widely used and 
practical methods for supervised learning.
Decision Trees are a non-parametric supervised learning method used for both 
classification and regression tasks. The goal is to create a model that 
predicts the value of a target variable by learning simple decision rules 
inferred from the data features.
The decision rules are generally in form of if-then-else statements. The deeper 
the tree, the more complex the rules and fitter the model.
"""
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Importing Libraries
"""The Tree Module imported here for classification is created by Anubhav Sharma"""

import pandas as pd
import random
random.seed(0)
from Tree.DecisionTreeFunction import DecisionTreeAlgorithm, calculate_accuracy
from Tree.Helperfunction import train_test_split
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Main Function
def main():
    
    # -------------------------------------------------------------------------
    # Dataset :- 1
    # -------------------------------------------------------------------------
    # Calling Data
    """
    Data is call for work. The Columns are selected here is according to the 
    BREAST CANCER DATASET from WINCONSIN Hospital Easily find on Kaggle 
    (https://www.kaggle.com/uciml/breast-cancer-wisconsin-data).
    Removing unnecessary columns from the dataset so that you won't face any 
    trouble regarding the dataset. I use Breast Cancer Dataset to train model 
    and predict whether the person is having cancer or not.
    """
    df1 = pd.read_csv('./Dataset/Breast Cancer/Breast_Cancer_Data.csv')
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # creating label at last of dataframe
    df1['label'] = df1.diagnosis
    df1['label'].replace(['B','M'], [0,1], inplace = True)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Droping unnecessory columns
    df1.drop([df1.columns[0], df1.columns[1], df1.columns[32]], axis = 1, 
             inplace = True)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Spliting into Train and Test DataFrame
    train_df1, test_df1 = train_test_split(df1, test_size = 0.25)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Classifiction Model Fitting
    tree1 = DecisionTreeAlgorithm(train_df1, mltask = 'classification', 
                                  max_depth = 7, min_samples = 8)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Calculating Accuracy
    # Training Data
    accuracy_train_data1 = calculate_accuracy(train_df1, tree1)
    print("Accuracy over Training Data 1 : {}".format(accuracy_train_data1))
    
    # Testing Data
    accuracy_test_data1 = calculate_accuracy(test_df1, tree1)
    print("Accuracy over Testing Data 1 : {}".format(accuracy_test_data1))
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Dataset :- 2
    # -------------------------------------------------------------------------
    # Calling Data
    """
    Data is call for work. The Columns are selected here is according to the 
    TITANIC TRAVELLERS DATASET. Easily find on Kaggle 
    (https://www.kaggle.com/hesh97/titanicdataset-traincsv).
    Removing unnecessary columns from the dataset so that you won't face any 
    trouble regarding the dataset. I use Titanic Dataset to train model and 
    predict whether the person is Survived or not.
    """
    df2 = pd.read_csv("./Dataset/Titanic Dataset/train.csv")
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Creating label at last of DataFrame
    df2["label"] = df2.Survived
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Dropping Unnecessory Columns
    df2.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace =  True)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Filling Nan Values of DataFrame
    median_age = df2.Age.median()
    
    df2 = df2.fillna({'Age' : median_age})
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Spliting into Train and Test DataFrame
    train_df2, test_df2 = train_test_split(df2, test_size = 0.25)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Classifiction Model Fitting
    tree2 = DecisionTreeAlgorithm(train_df2, mltask = 'classification', 
                                  max_depth = 30, min_samples = 38)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Calculating Accuracy
    # Training Data
    accuracy_train_data2 = calculate_accuracy(train_df2, tree2)
    print("Accuracy over Training Data 2 : {}".format(accuracy_train_data2))
    
    # Testing Data
    accuracy_test_data2 = calculate_accuracy(test_df2, tree2)
    print("Accuracy over Testing Data 2 : {}".format(accuracy_test_data2))
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#Calling Main Function
if __name__ == "__main__":
    
    # Calling Main Function
    main()
# -----------------------------------------------------------------------------