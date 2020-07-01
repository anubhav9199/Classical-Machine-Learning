# -----------------------------------------------------------------------------
# Decision tree regression from scratch over Breast Cancer Dataset Using Tree Module
"""
Decision tree builds regression or classification models in the form of a tree 
structure. It breaks down a dataset into smaller and smaller subsets while at 
the same time an associated decision tree is incrementally developed. The final
result is a tree with decision nodes and leaf nodes. A decision node 
(e.g., Cancer) has two or more branches (e.g., Benign and Malignant), each 
representing values for the attribute tested. Leaf node (e.g., Radius Mean,
Texture Mean) represents a decision on the numerical target. The topmost 
decision node in a tree which corresponds to the best predictor called root 
node. Decision trees can handle both categorical and numerical data.
"""
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Importing Libraries
"""
The code calls all the library use and the Tree module made by Anubhav Sharma.
"""
import pandas as pd
from Tree.DecisionTreeFunction import DecisionTreeAlgorithm, calculate_r_square
from Tree.Helperfunction import train_test_split
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Main Function
def main():
    
    # -------------------------------------------------------------------------
    # Calling Data
    """
    Data is call for work. The Columns are selected here is according to the 
    BREAST CANCER DATASET from WINCONSIN Hospital Easily find on Kaggle
    (www.kaggle.com).
    Removing unnecessary columns from the dataset so that you won't face any 
    trouble regarding the dataset. I use Breast Cancer Dataset to train model 
    and predict whether the person is having cancer or not.
    """
    df = pd.read_csv('Breast_Cancer_Data.csv')
    df['label'] = df.diagnosis
    df['label'].replace(['B','M'], [0,1], inplace = True)
    df.drop([df.columns[0], df.columns[1], df.columns[32]], axis = 1, 
            inplace =  True)
    
    # -------------------------------------------------------------------------
    # Spliting Data into Train And Test DataFrame
    train_df, test_df = train_test_split(df, test_size = 0.25)
    
    # -------------------------------------------------------------------------
    # Regression Fitting
    tree = DecisionTreeAlgorithm(df, mltask = 'regression')
    
    # -------------------------------------------------------------------------
    # Calculating R Square of the fitted data.
    # Training Data
    r_square_train = calculate_r_square(train_df, tree)
    print("R Square of Training Data : {}".format(r_square_train))
    
    # Testing Data
    r_square_test = calculate_r_square(test_df, tree)
    print("R Square of Testing Data : {}".format(r_square_test))
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#Calling Main Function
if __name__ == "__main__" :
    
    main()
# -----------------------------------------------------------------------------