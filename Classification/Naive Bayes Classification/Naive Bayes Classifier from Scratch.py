# Naive Bayes from Scratch over Breast Cancer Data
"""Naive Bayes is a simple technique for constructing classifiers: models that 
assign class labels to problem instances, represented as vectors of feature 
values, where the class labels are drawn from some finite set. There is not a 
single algorithm for training such classifiers, but a family of algorithms 
based on a common principle: all naive Bayes classifiers assume that the value 
of a particular feature is independent of the value of any other feature, 
given the class variable. For example, a fruit may be considered to be an apple 
if it is red, round, and about 10 cm in diameter. A naive Bayes classifier 
considers each of these features to contribute independently to the probability 
that this fruit is an apple, regardless of any possible correlations between 
the color, roundness, and diameter features."""

# Importing Libraries
import numpy as np
import pandas as pd

### Feature dictionary function
"""Function to create Dictionary of requested feature for a particular class 
using the relative frequency."""
def feature_dict(data,training_len,feature_name, cancer_class):
    p_values=[]
    feature_unique_values=data[feature_name].unique()
    cancer_train_data=data[data['diagnosis']==cancer_class].iloc[:training_len]
    cancer_unique=cancer_train_data[feature_name].unique()
    for feature in feature_unique_values:
        if feature in cancer_unique:
            rf=cancer_train_data[cancer_train_data[feature_name]==feature].shape[0]/cancer_train_data.shape[0]
        else:
            rf=1/(cancer_train_data.shape[0]+feature_unique_values.shape[0])
        p_values.append(rf)
    feature_dictionary=dict(zip(feature_unique_values,p_values))
    return feature_dictionary

### Posterior Probability Function
"""Function to calculate Posterior Probability."""
def calc_posterior(x,testing_b,testing_m):
    posterior_p = np.prod(testing_b.iloc[x])/(np.prod(testing_m.iloc[x])+np.prod(testing_b.iloc[x]))
    return posterior_p

def main():
    
    # Calling Data
    """Data is call for work. The Columns are selected here is according to the 
    BREAST CANCER DATASET from WINCONSIN Hospital Easily find on 
    Kaggle(www.kaggle.com)."""
    # Reading Data
    data = pd.read_csv("./Dataset/Breast Cancer Dataset/Breast_Cancer_Data.csv")
    
    # Droping Unnecessory Columns (Eg. Id, Unnamed 32, etc)
    data.drop([data.columns[0],data.columns[32]],axis = 1, inplace = True)
    
    # Differentiating data on basis of 'M' and 'B'
    m_data = data[data["diagnosis"]=='M']
    b_data = data[data["diagnosis"]=='B']
    
    # Splitting the training and testing Data
    # Length of the data to be used in training and testing
    training_len=int(0.7*data.shape[0])
    m_trainlen=int(training_len/2)
    b_trainlen=m_trainlen
    
    # Fetching out training andd testing data.
    # Training
    m_train=m_data.iloc[:m_trainlen]
    b_train=b_data.iloc[:b_trainlen]
    
    # Testing
    m_test=m_data.iloc[m_trainlen:]
    b_test=b_data.iloc[b_trainlen:]
    
    # Concating data.
    training=pd.concat([m_train,b_train])
    testing=pd.concat([m_test,b_test])
    
    # Creating Feature Dictionary for both the classes
    dictionary_m={}
    dictionary_b={}
    for i in data:
        if i=='diagnosis':
            continue
        dictionary_m[i]=feature_dict(data,training_len,i,'M')
        dictionary_b[i]=feature_dict(data,training_len,i,'B')
        
    # Creating Dataframe from Feature Dictionary
    testing['diagnosis'].replace(to_replace=['B','M'],value=[0,1],inplace=True)
    testing_b_pvalue=pd.DataFrame()
    testing_m_pvalue=pd.DataFrame()
    for i in data.columns[1:]:
        testing_m_pvalue[i]=testing[i].replace(to_replace=data[i].unique(),value=dictionary_m[i].values())
    for i in data.columns[1:]:
        testing_b_pvalue[i]=testing[i].replace(to_replace=data[i].unique(),value=dictionary_b[i].values())
    
    
    
    # Testing Data (Bening)
    # Testing on test data
    post_bclass = np.array(list(map(lambda x:calc_posterior(x,testing_b_pvalue,testing_m_pvalue),
                                    np.arange(0,testing.shape[0]))))
    acc_bening = (np.count_nonzero(np.equal(testing['diagnosis'],
                                            post_bclass.astype(int)))/testing.shape[0])*100
    print("Accuracy on testing Bening = ",acc_bening)
    
    # Testing Data (Malingnent)
    # Testing on test data
    post_mclass=np.array(list(map(lambda x:calc_posterior(x,testing_m_pvalue,testing_b_pvalue),
                                  np.arange(0,testing.shape[0]))))
    acc_malignent=(np.count_nonzero(np.equal(testing['diagnosis'],
                                             post_mclass.astype(int)))/testing.shape[0])*100
    print("Accuracy on testing Malignent =",acc_malignent)

# Main Function
if __name__ == "__main__":
    
    # Calling Main Function
    main()