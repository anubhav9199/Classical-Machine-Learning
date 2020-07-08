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
"""
The Random Forest Algorithm is use from tree module fully Created By 
Anubhav Sharma
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Tree.RandomForestFunction import RandomForestAlgorithm
from Tree.RandomForestFunction import RandomForestPrediction
from Tree.RandomForestFunction import RandomForestClassification
from Tree.HelperFunction import calculate_accuracy, train_test_split
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def main():
    # -------------------------------------------------------------------------
    # Dataset 1 (Continuous)
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
    cancer_data = pd.read_csv('./Dataset/Breast Cancer Dataset/Breast_Cancer_Data.csv')
    
    cancer_data['diagnosis'].replace(['B', 'M'], [0, 1], inplace = True)
    cancer_data['label'] = cancer_data.diagnosis
    
    cancer_df = {'radius_mean':[], 'texture_mean':[], 'perimeter_mean':[], 
                 'label':[]}
    cancer_df['radius_mean'] = cancer_data.radius_mean
    cancer_df['texture_mean'] = cancer_data.texture_mean
    cancer_df['perimeter_mean'] = cancer_data.perimeter_mean
    cancer_df['label'] = cancer_data.label
    
    cancer_df = pd.DataFrame(cancer_df)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Train Test Split
    train_cancer, test_cancer = train_test_split(cancer_df,test_size=0.25)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Random Forest Algorithm
    """
    Random Forest Algorithm from Random forest function created by 
    Anubhav Sharma
    """
    cancer_forest = RandomForestAlgorithm(train_cancer,n_trees=4,
                                          mltask='regression',
                                          n_bootstrap=350, n_feature=4,
                                          dt_max_depth=4)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Accuracy Prediction
    # Training Data
    prediction_cancer_train = RandomForestPrediction(train_cancer,cancer_forest)
    accuracy_cancer_train = calculate_accuracy(prediction_cancer_train,
                                               train_cancer.label)
    print("Accuracy over Breast Cancer Testing Data : {}".format(accuracy_cancer_train))
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Testing Data
    prediction_cancer_test = RandomForestPrediction(test_cancer,cancer_forest)
    accuracy_cancer_test = calculate_accuracy(prediction_cancer_test,
                                              test_cancer.label)
    print("Accuracy over Breast Cancer Testing Data : {}".format(accuracy_cancer_test))
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # DataSet 2 (Catagorical Data)
    # -------------------------------------------------------------------------
    # Calling Data
    """
    Data is call for work. The Columns are selected here is according to the 
    TITANIC DATASET. Easily find on Kaggle(www.kaggle.com).
    Removing unnecessary columns from the dataset so that you won't face any 
    trouble regarding the dataset. I use Titanic Dataset to train model and 
    predict whether the person is having cancer or not.
    """
    titanic_df = pd.read_csv('./Dataset/Titanic Dataset/train.csv')
    titanic_df['label'] = titanic_df.Survived
    
    titanic_df.drop(['PassengerId','Survived','Name','Ticket','Cabin','Embarked']
                    ,axis=1,inplace=True)
    
    median_age = titanic_df.Age.median()
    titanic_df = titanic_df.fillna({'Age' : median_age})
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Train Test Split
    train_titanic, test_titanic = train_test_split(titanic_df,test_size=0.25)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Random Forest Algorithm
    """
    Random Forest Algorithm from Random forest function created by Anubhav Sharma
    """
    titanic_forest = RandomForestAlgorithm(train_titanic,n_trees=10,
                                           mltask='regression',n_bootstrap=500,
                                           n_feature=4,dt_max_depth=10)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Accuracy Prediction
    # Training Data
    prediction_titanic_train = RandomForestPrediction(train_titanic,titanic_forest)
    accuracy_titanic_train = calculate_accuracy(prediction_titanic_train,
                                                train_titanic.label)
    print("Accuracy over Breast Cancer Testing Data : {}".format(accuracy_titanic_train))
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Testing Data
    prediction_titanic_test = RandomForestPrediction(test_titanic,titanic_forest)
    accuracy_titanic_test = calculate_accuracy(prediction_titanic_test,
                                               test_titanic.label)
    print("Accuracy over Breast Cancer Testing Data : {}".format(accuracy_titanic_test))
    # -------------------------------------------------------------------------

    
# -----------------------------------------------------------------------------
# Calling Main Function    
if __name__ == "__main__" :
    
    # Calling Main Function
    main()
# -----------------------------------------------------------------------------