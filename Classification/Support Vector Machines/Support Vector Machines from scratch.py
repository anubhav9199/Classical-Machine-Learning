# Support Vector Machine from Scratch using SMO
"""Support Vector Machines are a type of supervised machine learning algorithm 
that provides analysis of data for classification and regression analysis. 
While they can be used for regression, SVM is mostly used for classification. 
We carry out plotting in the n-dimensional space. The value of each feature is 
also the value of the specified coordinate. Then, we find the ideal hyperplane 
that differentiates between the two classes.
These support vectors are the coordinate representations of individual 
observation. It is a frontier method for segregating the two classes.
This machine learning model is able to generalise between two different 
classes if the set of labelled data is provided in the training set to the 
algorithm. The main function of the SVM is to check for that hyperplane that 
is able to distinguish between the two classes.
I use Sequential Minimal Optimization Algorithm for the fast result over SVM 
Model. John Platt from Microsoft gave this amazing technique to train the model 
in a fast way. In this SMO actually check some of the lambdas of data which is 
find by the SVM over KKT Conditions. If any of the lambdas fit all the 
conditions then it will be treated as Support Vector for the training and 
fixing the Hyperplane."""

# Importing libraries
import pandas as pd
import numpy as np

# Theta Function
"""Funtion to return thetas after the SVM trains itself on training data."""
def thetas(lambdas,C_train,x):
    
    a=lambdas*C_train.T
    
    theta=np.matmul(x.T,a.T)
    
    neg_x=x[np.where(C_train==-1)[0]]
    pos_x=x[np.where(C_train==1)[0]]
    
    pos=np.min(1-np.matmul(theta.T,pos_x.T))
    neg=np.max(-1-np.matmul(theta.T,neg_x.T))
    
    theta0=(pos+neg)/2
    
    return theta,theta0

# Error Function
"""Funtion to calculate Error between new lambdas and old lambdas."""
def E(i,b,C_train,lambdas,x):
    
    fx=np.sum((C_train*lambdas)*np.matmul(x,x[i].T))+b
    
    return fx-C_train[i]

# L and H Bound Function
"""Funtion to find upper and lower bounds which don't voilate KKT condition. L 
is lower bound and H is upper bound."""
def LnH(i,j,regularize,C_train,lambdas):
    
    if C_train[i]!=C_train[j]:
        L=max(0, lambdas[j]-lambdas[i])
        H=min(regularize,regularize+lambdas[j]-lambdas[i])
    
    if C_train[i]==C_train[j]:
        L=max(0,lambdas[i]+lambdas[j]-regularize)
        H=min(regularize,lambdas[i]+lambdas[j])
    
    return L,H

# Compute Eta Function
"""Funtion to return inner product (or implement some kernel funtion K)"""
def compute_eta(i,j,x):
    
    return (2*np.matmul(x[i],x[j].T)-np.matmul(x[i],x[i].T)-np.matmul(x[j],x[j].T))

# Compute B-Threshold Function
"""Funtion to compute threshold i.e. B which is compute after finding the 
lambdas so that it can satisfied the vectors."""
def compute_b(i,j,b,E1,E2,R,C_train,lambdas,lambdas_old,x):
    
    # Finding y*(alphai - aplhai_old) & y(alphaj - alphaj_old)
    I=C_train[i]*(lambdas[i]-lambdas_old[i])
    J=C_train[j]*(lambdas[j]-lambdas_old[j])
    
    # Finding b1 and b2
    b1=b-E1-(I*np.matmul(x[i],x[i].T))-(J*np.matmul(x[i],x[j].T))
    b2=b-E2-(I*np.matmul(x[i],x[j].T))-(J*np.matmul(x[j],x[j].T))
    
    if lambdas[i]>0 and lambdas[i]<R:
        b=b1
    
    elif lambdas[j]>0 and lambdas[j]<R:
        b=b2
    
    else:
        b=(b1+b2)/2
    
    return b

# Cliping Function
"""Function to clip the value of 'Lagrange Multiplier' if they voilate KKT 
condition"""
def clip(H,L,j,lambdas):
    
    if lambdas[j]>H:
        lambdas[j]=H
    
    elif lambdas[j]<L:
        lambdas[j]=L
    
    elif L<=lambdas[j]<=H:
        lambdas[j]=lambdas[j]

# Calculate J Function
"""Funtion to pick second Lagrange multiplier which results the max value of 
Error with the first Lagrange Multiplier"""
def second_lambda(i,x,b,C_train,lambdas):
    elist=[]
    for k in range(0,x.shape[0]):
        e=E(k,b,C_train,lambdas,x)
        elist.append(abs(e-E(i,b,C_train,lambdas,x)))
    new_j=np.argmax(elist)
    return new_j

def main():
    # Calling Data
    """Data is call for work. The Columns are selected here is according to the 
    BREAST CANCER DATASET from WINCONSIN Hospital Easily find on 
    Kaggle(www.kaggle.com).
    Creating Dataset from original columns from the dataset so that you won't face 
    any trouble regarding the dataset. I use Breast Cancer Dataset to train model 
    and predict whether the person is having cancer or not."""
    data=pd.read_csv("./Dataset/Breast Cancer Dataset/Breast_Cancer_Data.csv")
    
    # Taking Out Labels
    C=(data['diagnosis'])
    C.replace(to_replace=['B','M'],value=[1,-1],inplace=True)
    C=np.array(C).reshape(569,1)
    
    # Data Preprocessing removing Unnecessory columns
    data.drop([data.columns[0],data.columns[1],data.columns[32]],axis = 1, inplace = True)
    data=(data-np.mean(data,axis=0))/np.std(data,axis=0)
    
    # Splitting training and testing data
    training_len=0.75*data.shape[0]
    train_data=data.iloc[-int(training_len):]
    test_data=data.iloc[:data.shape[0]-int(training_len)]
    x=np.array(train_data)
    
    C_train=C[:int(training_len)]
    C_test=C[int(training_len):]
    
    # Initializing parameters for Sequencial Minimal Optimization
    max_passes=4
    passes=0
    tol=10**(-1)
    R=x.shape[0]
    
    lambdas=np.zeros((C_train.shape[0]))
    #lambdas[np.where(lambdas<0)[0]]=0
    lambdas_old=np.zeros((C_train.shape[0]))
    b=0
    
    # Sequential Minimal Optimization (Refrenced from : CS229 Simplified SMO)
    while passes < max_passes:
        changed_alphas = 0
        for i in range(0,x.shape[0]):
            E1=E(i,b,C_train,lambdas,x)
            if (C_train[i]*E1 < -tol and lambdas[i] < R) or (C_train[i]*E1 > tol and lambdas[i] > 0):
                j=second_lambda(i,x,b,C_train,lambdas)
                E2=E(j,b,C_train,lambdas,x)
                lambdas_old[i]=lambdas[i]
                lambdas_old[j]=lambdas[j]
                L,H=LnH(i,j,R,C_train,lambdas)
                if L == H:
                    continue
                eta=compute_eta(i,j,x)
                if eta >= 0:
                    continue
                lambdas[j]=lambdas[j]+(C_train[j]*(int(E2-E1)))/eta
                clip(H,L,j,lambdas)
                if (abs(lambdas[j] - lambdas_old[j]) < 10**(-5)):
                    continue
                lambdas[i]=lambdas[i]-(C_train[i]*C_train[j]*(lambdas[j]-lambdas_old[j]))
                b=compute_b(i,j,b,E1,E2,R,C_train,lambdas,lambdas_old,x)
                changed_alphas = changed_alphas + 1
        if changed_alphas == 0:
            passes=passes + 1
        else:
            passes=0
        print(passes)
    
    # Number of support vectors
    np.count_nonzero(lambdas)
    # Getting the parameters for Decision Boundary
    t,t0=thetas(lambdas,C_train,x)
    
    # Testing Accuracy
    y=np.array(test_data)
    predicted_labels=np.matmul(y,t)+t0
    predicted_labels[np.where(predicted_labels>0)[0]]=1
    predicted_labels[np.where(predicted_labels<=0)[0]]=-1
    o=np.count_nonzero(np.equal(C_test,predicted_labels))
    test_accuracy=(o/C_test.shape[0])*100
    print("Testing accuracy : ",test_accuracy)
    
    # Training Accuracy
    training_label=np.matmul(x,t)+t0
    training_label[np.where(training_label>0)[0]]=1
    training_label[np.where(training_label<0)[0]]=-1
    train_accuracy = np.count_nonzero(np.equal(C_train,training_label))/C_train.shape[0]*100
    print("Training accuracy : ",train_accuracy)


# Calling Main Function
if __name__ == "__main__":
    
    # Calling Main Function
    main()