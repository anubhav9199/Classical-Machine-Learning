# -----------------------------------------------------------------------------
# Support Vector Regression from Scratch using Support Vector Machine Module
"""
Support Vector Regression(SVR) is quite different than other Regression models. 
It uses the Support Vector Machine(SVM, a classification algorithm) algorithm 
to predict a continuous variable. While other linear regression models try to 
minimize the error between the predicted and the actual value, Support Vector 
Regression tries to fit the best line within a predefined or threshold error 
value. What SVR does in this sense, it tries to classify all the prediction 
lines in two types, ones that pass through the error boundary( space separated 
by two parallel lines) and ones that don’t. Those lines which do not pass the 
error boundary are not considered as the difference between the predicted 
value and the actual value has exceeded the error threshold, 𝞮(epsilon). The 
lines that pass, are considered for a potential support vector to predict the 
value of an unknown. The following illustration will help you to grab this 
concept.
"""
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Importing Libraries
"""
Importing Support_Vector_Machine Module made by Anubhav Sharma
"""
import pandas as pd
from Support_Vector_Machine.SupportVectorMachine import SVR
from Support_Vector_Machine.HelperFunction import confusionMatrix 
from Support_Vector_Machine.HelperFunction import train_test_split
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Main Function
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
    cancer_df['diagnosis'].replace(['B', 'M'], [0, 1], inplace = True)
    cancer_df['labels'] = cancer_df.diagnosis
    cancer_df.drop([cancer_df.columns[0], cancer_df.columns[1], 
                    cancer_df.columns[32]], axis = 1, inplace = True)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Removing data and labels for the model to fit.
    cancer_data = cancer_df.iloc[:, 2:3].values
    cancer_label = cancer_df.iloc[:, -1].values
    cancer_label = cancer_label.reshape(cancer_label.shape[0],1)
    cancer_data = pd.DataFrame(cancer_data)
    cancer_label = pd.DataFrame(cancer_label)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Train Test Split
    train_cancer, test_cancer = train_test_split(cancer_data, test_size = 0.25)
    train_label, test_label = train_test_split(cancer_label, test_size = 0.25)
    train_cancer = train_cancer.values
    test_cancer = test_cancer.values
    train_label = train_label.values
    test_label = test_label.values
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Creating and Fitting Model
    SupportVectorRegression = SVR()
    SupportVectorRegression.fit(train_cancer, train_label)
    # -------------------------------------------------------------------------
    
    # Prediction Over Model
    # -------------------------------------------------------------------------
    # Training Data
    train_predict = SupportVectorRegression.predict(train_cancer)
    train_cm, train_accuracy, train_precision , train_recall, train_fm = confusionMatrix(train_label,train_predict)
    
    print("-----------------------------------------------------")
    print("Confusion Matrix : {}\nAccuracy over Training Data : {}\nPrecision : {}\nRecall : {}\nFM : {}".format(train_cm, train_accuracy, train_precision, train_recall, train_fm))
    print("-----------------------------------------------------")
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Testing Data
    test_predict = SupportVectorRegression.predict(test_cancer)
    test_cm, test_accuracy, test_precision , test_recall, test_fm = confusionMatrix(test_label,test_predict)
    
    print("-----------------------------------------------------")
    print("Confusion Matrix : {}\nAccuracy over Testing Data : {}\nPrecision : {}\nRecall : {}\nFM : {}".format(test_cm, test_accuracy, test_precision , test_recall, test_fm))
    print("-----------------------------------------------------")
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Calling Main Function
if __name__ == "__main__" :
    
    # Calling Main Function
    main()
# -----------------------------------------------------------------------------