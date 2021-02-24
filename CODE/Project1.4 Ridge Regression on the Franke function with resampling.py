# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 22:35:03 2021

@author: Alma Kurmanova and Yash Kumar
"""

"""Project 1 - Part (d) - Ridge Regression on the Franke function with resampling"""

#%%
"""Import all necessary packages"""

import numpy as np                         
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.model_selection import KFold                #for KFold cross-validation
from sklearn.model_selection import train_test_split as splitter 
from sklearn.utils import resample

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
z1=FrankeFunction(x,y)

#%%
"""Function to make design matrix"""

def DesMatrix(deg,data1,data2):
    X=np.ones((len((data1)),1))
    for i in range(deg+1):
        if (i>0):
            for j in range(i+1):
                temp=(data1**j)*(data2**(i-j))
                X=np.hstack((X,temp))
    return X

#%%
"""Define parameters"""

polynomial=np.arange(0,order+1,1)      #define polynomial degree array between 0 and 5 
lamda=np.logspace(-5,5,11)             #define the hyperparameters

#%%
"""Declare arrays to store MSE and r2-score for bootstrap"""

mse_ridge_boot=np.zeros((polynomial.shape[0],lamda.shape[0],2))
r2_ridge_boot=np.zeros((polynomial.shape[0],lamda.shape[0],2))

#%%
"""Split data into training and testing data and perform regression with bootstrap"""

n_bootstrap=500  #bootstrap

for i in range(order+1):
    for j in range(len(lamda)):
        X=DesMatrix(polynomial[i],x,y)
        X_train,X_test,z_train,z_test=splitter(X,z,test_size=0.2,random_state=12)
        z_pred_train=np.zeros((len(z_train),n_bootstrap))
        z_pred_test=np.zeros((len(z_test),n_bootstrap))
        for k in range(n_bootstrap):
            X_,z_=resample(X_train,z_train)
            clf=skl.Ridge(alpha=lamda[j],fit_intercept=False).fit(X_,z_)
            z_pred_test[:,k]=clf.predict(X_test).flatten()
            z_pred_train[:,k]=clf.predict(X_train).flatten()
        mse_ridge_boot[i,j,0]=MSE(z_train,np.mean(z_pred_train,axis=1,keepdims=True))
        mse_ridge_boot[i,j,1]=MSE(z_test,np.mean(z_pred_test,axis=1,keepdims=True))
        r2_ridge_boot[i,j,0]=R2(z_train,np.mean(z_pred_train,axis=1,keepdims=True))
        r2_ridge_boot[i,j,1]=R2(z_test,np.mean(z_pred_test,axis=1,keepdims=True))

#%%
"""Declare arrays to store MSE and r2-score for CV"""

mse_ridge_CV=np.zeros((polynomial.shape[0],lamda.shape[0],2))
r2_ridge_CV=np.zeros((polynomial.shape[0],lamda.shape[0],2))

#%%
"""Split data into training and testing data and perform regression with CV"""

k=10                                    #define number of folds
kfold=KFold(n_splits=k)                #introduce k-fold cross-validation

for i in range(order+1):               #loop over model complexity       
    M=DesMatrix(polynomial[i],x,y)     #construct design matrix 
    for j in range(len(lamda)):
        u=0
        for train,test in kfold.split(M):  #split the design matrix and array into train and test
            M_train=M[train]
            z_train=z[train].reshape(-1,1)                 
            M_test=M[test]
            z_test=z[test].reshape(-1,1)
            zpred_train=np.zeros((len(z_train),k))  #initialise matrix to store model predicted data for each fold
            zpred_test=np.zeros((len(z_test),k))       
            clf=skl.Ridge(alpha=lamda[j],fit_intercept=False).fit(M_train,z_train)
            zpred_test[:,u]=clf.predict(M_test).flatten()
            zpred_train[:,u]=clf.predict(M_train).flatten()
            u+=1
        mse_ridge_CV[i,j,0]=MSE(z_train,np.mean(zpred_train,axis=1,keepdims=True))    #calculate train MSE
        mse_ridge_CV[i,j,1]=MSE(z_test,np.mean(zpred_test,axis=1,keepdims=True))      #calculate test MSE
        r2_ridge_CV[i,j,0]=R2(z_train,np.mean(zpred_train,axis=1,keepdims=True))      #calculate train R2
        r2_ridge_CV[i,j,1]=R2(z_test,np.mean(zpred_test,axis=1,keepdims=True))        #calculate test R2
    
#%%
"""Plot results"""

plt.figure()
plt.plot(polynomial,mse_ridge_boot[:,3,0],'r--',label='Train data, $\u03BB=10^{-2}, bootstrap$')
plt.plot(polynomial,mse_ridge_CV[:,3,0],'r-',label='Train data, $\u03BB=10^{-2}, CV$')
plt.plot(polynomial,mse_ridge_boot[:,3,1],'b--',label='Test data, $\u03BB=10^{-2}, bootstrap$')
plt.plot(polynomial,mse_ridge_CV[:,3,1],'b-',label='Test data, $\u03BB=10^{-2}, CV$')
plt.plot(polynomial,mse_ridge_boot[:,6,0],'g--',label='Train data, $\u03BB=10, bootstrap$')
plt.plot(polynomial,mse_ridge_CV[:,6,0],'g-',label='Train data, $\u03BB=10, CV$')
plt.plot(polynomial,mse_ridge_boot[:,6,1],'m--',label='Test data, $\u03BB=10, bootstrap$')
plt.plot(polynomial,mse_ridge_CV[:,6,1],'m-',label='Test data, $\u03BB=10, CV$')
plt.xlabel('Model Complexity',fontweight='bold')
plt.ylabel('MSE',fontweight='bold')
plt.title('MSE, 10-fold CV, n=1000, boot=500',fontweight='bold')
plt.grid()
plt.legend(loc='best',fontsize='x-small')
plt.show()

plt.figure()
plt.plot(polynomial,r2_ridge_boot[:,3,0],'r--',label='Train data, $\u03BB=10^{-2}, bootstrap$')
plt.plot(polynomial,r2_ridge_CV[:,3,0],'r-',label='Train data, $\u03BB=10^{-2}, CV$')
plt.plot(polynomial,r2_ridge_boot[:,3,1],'b--',label='Test data, $\u03BB=10^{-2}, bootstrap$')
plt.plot(polynomial,r2_ridge_CV[:,3,1],'b-',label='Test data, $\u03BB=10^{-2}, CV$')
plt.plot(polynomial,r2_ridge_boot[:,6,0],'g--',label='Train data, $\u03BB=10, bootstrap$')
plt.plot(polynomial,r2_ridge_CV[:,6,0],'g-',label='Train data, $\u03BB=10, CV$')
plt.plot(polynomial,r2_ridge_boot[:,6,1],'m--',label='Test data, $\u03BB=10, bootstrap$')
plt.plot(polynomial,r2_ridge_CV[:,6,1],'m-',label='Test data, $\u03BB=10, CV$')
plt.xlabel('Model Complexity',fontweight='bold')
plt.ylabel('$\mathbf{r^{2}}$',fontweight='bold')
plt.title('$\mathbf{r^{2}}$, 10-fold CV, n=1000, boot=500',fontweight='bold')
plt.grid()
plt.legend(loc='best',fontsize='x-small')
plt.show()

#%%
"""Define function to calculate bias, variance and error"""

def metrics2(y_true,y_pred):
    error=np.mean(np.mean((y_true-y_pred)**2,axis=1,keepdims=True))
    bias=np.mean((y_true-np.mean(y_pred,axis=1,keepdims=True))**2)
    variance=np.var(np.mean(y_pred,axis=1,keepdims=True))
    return error,bias,variance

#%%
"""Bias-Variance Analysis"""

bias=np.zeros((lamda.shape[0],2))
variance=np.zeros((lamda.shape[0],2))
error=np.zeros((lamda.shape[0],2))

n_bootstrap=500  #bootstrap

for j in range(len(lamda)):
    X=DesMatrix(4,x,y)
    X_train,X_test,z_train,z_test=splitter(X,z,test_size=0.2,random_state=12)
    z_pred_train=np.zeros((len(z_train),n_bootstrap))
    z_pred_test=np.zeros((len(z_test),n_bootstrap))
    for k in range(n_bootstrap):
        X_,z_=resample(X_train,z_train)
        clf=skl.Ridge(alpha=lamda[j],fit_intercept=False).fit(X_,z_)
        z_pred_test[:,k]=clf.predict(X_test).flatten()
        z_pred_train[:,k]=clf.predict(X_train).flatten()
    error[j,0]=metrics2(z_train,z_pred_train)[0]
    error[j,1]=metrics2(z_test,z_pred_test)[0]
    bias[j,0]=metrics2(z_train,z_pred_train)[1]
    bias[j,1]=metrics2(z_test,z_pred_test)[1]
    variance[j,0]=metrics2(z_train,z_pred_train)[2]
    variance[j,1]=metrics2(z_test,z_pred_test)[2]
        
#%%
"""Plot bias-variance"""

plt.figure()
plt.plot(lamda,bias[:,0],'b--',label='Bias train data')
plt.plot(lamda,bias[:,1],'b-',label='Bias test data')
plt.plot(lamda,variance[:,0],'g--',label='Var train data')
plt.plot(lamda,variance[:,1],'g-',label='Var test data')    
# plt.plot(lamda,error[:,0],'m--',label='Error train data')
# plt.plot(lamda,error[:,1],'m-',label='Error test data')
plt.xlabel('$\u03BB$',fontweight='bold')
plt.xscale('log')
plt.ylabel('Errors',fontweight='bold')
plt.title('Bias-Variance Analysis, bootstrap=500',fontweight='bold')
plt.grid()
plt.legend(loc='best')
plt.show()

