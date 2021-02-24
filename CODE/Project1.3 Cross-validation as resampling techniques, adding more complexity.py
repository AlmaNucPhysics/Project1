# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:16:05 2021

@author: Alma Kurmanova and Yash Kumar 
"""

"""Project 1 - Part (c) - Cross-validation applied to OLS regression on Franke Function"""

#%%
""""Import all necessary packages"""

import numpy as np                                       #for numerical operations
import matplotlib.pyplot as plt                          #for plotting purposes
import sklearn.linear_model as skl                       #for regression using scikit-learn
from sklearn.model_selection import KFold                #for KFold cross-validation
from sklearn.preprocessing import StandardScaler         #for scaling data   
from sklearn.metrics import mean_squared_error as MSE    #for calculating MSE using scikit-learn
from sklearn.metrics import r2_score as R2               #for calculating R2 using scikit-learn

#%%
"""Set the parameters"""

n=1000             #Define total number of data points
order=5            #Define max polynomial degree 
noise=0.1          #Define noise in the system  

"""Generate the data and define Franke Function"""

x=np.linspace(0,1,n).reshape(-1,1)      #define x array of n points between 0 and 1
y=np.linspace(0,1,n).reshape(-1,1)      #define y array of n points between 0 and 1

def FrankeFunction(x,y):
    term1=3/4*np.exp(-1/4*((9*x-2)**2)-1/4*((9*y-2)**2))
    term2=3/4*np.exp(-(1/49*(9*x-2)**2)-((9*y+1)/10))
    term3=1/2*np.exp(-1/4*((-9*x-7)**2)-1/4*((9*y-3)**2))
    term4=1/5*np.exp(-((9*x-4)**2)-((9*y-7)**2))
    return term1+term2+term3-term4

z=FrankeFunction(x,y)+np.random.normal(0,noise,x.shape)  #calculate z array with noise

#%%
"""Function to make design matrix"""

def DesMatrix(deg,data1,data2):                  #definition for function to make design matrix X
    X=np.ones((len((data1)),1))                  #default design matrix with ones column
    for i in range(deg+1):                       #loop to generate terms with powers of x and y
        if (i>0):
            for j in range(i+1):
                temp=(data1**j)*(data2**(i-j))
                X=np.hstack((X,temp))            #add terms to design matrix X
    return X

#%%
"""Design Martix"""

polynomial=np.arange(0,order+1,1)      #define polynomial degree array between 0 and 5          
k=10                                    #define number of folds
kfold=KFold(n_splits=k)                #introduce k-fold cross-validation

MSE_train=np.zeros(polynomial.shape)   #initialise array to store MSE values of training data
MSE_test=np.zeros(polynomial.shape)    #initialise array to store MSE values of testing data

R2_train=np.zeros(polynomial.shape)   #initialise array to store r2 values of training data
R2_test=np.zeros(polynomial.shape)    #initialise array to store r2 values of testing data
  
Error_CV_train=np.zeros(polynomial.shape)   #initialise array to store error values of training data  
Error_CV_test=np.zeros(polynomial.shape)    #initialise array to store error values of testing data
     
Var_train=np.zeros(polynomial.shape)        #initialse array to store variance values of training data
Var_test=np.zeros(polynomial.shape)         #initialise array to store variance values of testing data

Bias_train=np.zeros(polynomial.shape)       #initialise array to store bias values of training data
Bias_test=np.zeros(polynomial.shape)        #initialise array to store bias values of testing data 
 
for i in range(order+1):               #loop over model complexity                                            
    j=0
    M=DesMatrix(i,x,y)                 #construct design matrix 
    # scaler=StandardScaler()
    # scaler.fit(M)
    # M=scaler.transform(M)
    for train,test in kfold.split(M):  #split the design matrix and array into train and test
        M_train=M[train]
        z_train=z[train].reshape(-1,1)                 
        M_test=M[test]
        z_test=z[test].reshape(-1,1)
        scaler=StandardScaler()                 #initialise scaler
        scaler.fit(M_train)                     #evaluate mean and SD of data for normalisation
        M_train=scaler.transform(M_train)       #normalise elements of training matrix using Gaussian dist
        M_test=scaler.transform(M_test)         #normalise elements of testing matrix using Gaussian dist
        z_train=np.reshape(z_train,(len(z_train),1))
        z_test=np.reshape(z_test,(len(z_test),1))
        zpred_train=np.zeros((len(z_train),k))  #initialise matrix to store model predicted data for each fold
        zpred_test=np.zeros((len(z_test),k))       
        clf=skl.LinearRegression(fit_intercept=True).fit(M_train,z_train)  #perform OLS regression using scikit-learn  
        zpred_train[:,j]=clf.predict(M_train).flatten()  #store predicted training values in declared matrix
        zpred_test[:,j]=clf.predict(M_test).flatten()    #store predicted testing values in declared matrix
        j+=1
    Error_CV_train[i]=np.mean(np.mean((z_train-zpred_train)**2,axis=1,keepdims=True)) #calculate train error
    Error_CV_test[i]=np.mean(np.mean((z_test-zpred_test)**2,axis=1,keepdims=True))    #calculate test error
    Bias_train[i]=np.mean((z_train-np.mean(zpred_train,axis=1,keepdims=True))**2)     #calculate train bias
    Bias_test[i]=np.mean((z_test-np.mean(zpred_test,axis=1,keepdims=True))**2)        #calculate test bias
    Var_train[i]=np.mean(np.var(zpred_train,axis=1,keepdims=True))                #calculate train variance
    Var_test[i]=np.mean(np.var(zpred_test,axis=1,keepdims=True))                  #calculate test variance
    MSE_train[i]=MSE(z_train,np.mean(zpred_train,axis=1,keepdims=True))    #calculate train MSE
    MSE_test[i]=MSE(z_test,np.mean(zpred_test,axis=1,keepdims=True))       #calculate test MSE
    R2_train[i]=R2(z_train,np.mean(zpred_train,axis=1,keepdims=True))      #calculate train R2
    R2_test[i]=R2(z_test,np.mean(zpred_test,axis=1,keepdims=True))        #calculate test R2
    
#%%
"""Plot results"""

# plt.figure()
# plt.plot(polynomial,Error_CV_train,'m--',label='Error train data')
# plt.plot(polynomial,Error_CV_test,'m-',label='Error test data')
# plt.plot(polynomial,Bias_train,'b--',label='Bias train data')
# plt.plot(polynomial,Bias_test,'b-',label='Bias test data')
# plt.plot(polynomial,Var_train,'g--',label='Variance train data')
# plt.plot(polynomial,Var_test,'g-',label='Variance test data')
# plt.xlabel('Model Complexity',fontweight='bold')
# plt.ylabel('Errors',fontweight='bold')
# plt.title('Noise SD=0.1, 5-fold CV, n=1000',fontweight='bold')
# plt.grid()
# plt.legend(loc='best')
# plt.show()

plt.figure()
plt.plot(polynomial,MSE_train,'b--',label='MSE train data')
plt.plot(polynomial,MSE_test,'b-',label='MSE test data')
plt.xlabel('Model Complexity',fontweight='bold')
plt.ylabel('MSE',fontweight='bold')
plt.title('Noise SD=0.1, 10-fold CV, n=1000',fontweight='bold')
plt.grid()
plt.legend(loc='best')
plt.show()

# plt.figure()
# plt.plot(polynomial,R2_train,'g--',label='$r^{2}$ train data')
# plt.plot(polynomial,R2_test,'g-',label='$r^{2}$ test data')
# plt.xlabel('Model Complexity',fontweight='bold')
# plt.ylabel('$r^{2}$',fontweight='bold')
# plt.title('Noise SD=0.1, 5-fold CV, n=1000',fontweight='bold')
# plt.grid()
# plt.legend(loc='best')
# plt.show()

