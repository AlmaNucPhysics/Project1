# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:00:36 2021

@author: Alma Kurmanova and Yash Kumar
"""

"""Project 1 - Part (a) - Ordinary Least Squares Regression on Franke Function"""

#%%
"""Import all necessary packages"""

import numpy as np                                                 #for numerical operations
import matplotlib.pyplot as plt                                    #for plotting operations
from sklearn.model_selection import train_test_split as splitter   #for splitting data into train and test set 
from sklearn.metrics import mean_squared_error as MSE_score        #for calculating MSE values
from sklearn.metrics import r2_score as R2                         #for calculating R2 values   
from sklearn.utils import resample                                 #for bootstrap resampling
from matplotlib import cm                                          #for colormap
from mpl_toolkits.mplot3d import Axes3D

#%%
"""Define Franke Function"""

def FrankeFunction(x,y):                                       #definition for Franke function
    term1=(3/4)*np.exp(-(1/4)*(9*x-2)**2-(1/4)*(9*y-2)**2)
    term2=(3/4)*np.exp(-(1/49)*(9*x+1)**2-(1/10)*(9*y+1))
    term3=(1/2)*np.exp(-(1/4)*(9*x-7)**2-(1/4)*(9*y-3)**2)
    term4=(1/5)*np.exp(-(9*x-4)**2-(9*y-7)**2)
    return term1+term2+term3-term4

#%%
"""Define data for plotting function"""

x_plot=np.arange(0,1,0.05)                   #define x_plot array 
y_plot=np.arange(0,1,0.05)                   #define y_plot array

X,Y=np.meshgrid(x_plot,y_plot)               #define meshgrid for 2D plotting

z_plot=FrankeFunction(X,Y)                   #obtain data values for Franke function

#%%
"""Plot Franke Function"""

fig=plt.figure()                             #define new figure window
ax=fig.gca(projection='3d')                  #extract axis element

surf=ax.plot_surface(X,Y,z_plot,cmap=cm.coolwarm,linewidth=0,antialiased=False)   #plot Franke function
plt.xlabel('X')
plt.ylabel('Y')
fig.colorbar(surf,shrink=0.5,aspect=5)       #colorbar properties
plt.show()

del x_plot,y_plot,z_plot,X,Y                 #clear memory

#%%
"""Define dataset for OLS fitting"""

x=np.linspace(0,1,5001).reshape(-1,1)         #define x array of 5001 points between 0 and 1
y=np.linspace(0,1,5001).reshape(-1,1)         #define y array of 5001 points between 0 and 1

z=FrankeFunction(x,y)+np.random.normal(0,0.1,x.shape) #calculate z array with noise
#z1=FrankeFunction(x,y)

#%%
"""Define array for order of polynomial"""

order=np.arange(0,6,1)                      #define order array between 0 and 5

#%%
"""Define function for design matrix"""

def DesMatrix(degree,data1,data2):           #definition for function to make design matrix X
    X=np.ones((len(data1),1))                #default design matrix with ones column 
    for i in range(degree+1):
        if (i>0):
            for j in range(i+1):
                temp=(data1**j)*(data2**(i-j))
                X=np.hstack((X,temp))
    return X

#%%
"""Define function for OLS regression"""

def OLS(X_,X_train,X_test,y_):            #definition for function to perform OLS fitting
    beta=np.linalg.pinv(X_)@y_
    y_train=X_train@beta
    y_test=X_test@beta
    return y_train,y_test

#%%
"""Define function to calculate MSE and r2 scores"""

def metrics1(y_true,y_pred):
    mse=MSE_score(y_true,y_pred)
    # mse=np.mean((y_true-y_pred)**2)
    r2=R2(y_true,y_pred)
    # r2=1-np.sum((y_true-y_pred)**2)/np.sum((y_true-np.mean(y_true))**2)
    return mse,r2

#%%
"""Define function to calculate bias, variance and error"""

def metrics2(y_true,y_pred):
    error=np.mean(np.mean((y_true-y_pred)**2,axis=1,keepdims=True))
    bias=np.mean((y_true-np.mean(y_pred,axis=1,keepdims=True))**2)
    variance=np.mean(np.var(y_pred,axis=1,keepdims=True))
    return error,bias,variance

#%%
"""Declare arrays to store MSE and r2-score and create empty list for parameter variances (no scaling)"""

MSE=np.zeros((order.shape[0],2))
r2=np.zeros((order.shape[0],2))
bias=np.zeros((order.shape[0],2))
variance=np.zeros((order.shape[0],2))
error=np.zeros((order.shape[0],2))

#%%
"""Split data into training and testing data and perform regression with bootstrap"""

n_bootstrap=500
    
for i in range(len(order)):
    X=DesMatrix(order[i],x,y)
    X_train,X_test,z_train,z_test=splitter(X,z,test_size=0.2,random_state=12)
    z_pred_train=np.zeros((len(z_train),n_bootstrap))
    z_pred_test=np.zeros((len(z_test),n_bootstrap))
    for j in range(n_bootstrap):
        X_,z_=resample(X_train,z_train)
        z_pred_train[:,j]=OLS(X_,X_train,X_test,z_)[0].flatten()
        z_pred_test[:,j]=OLS(X_,X_train,X_test,z_)[1].flatten()
    MSE[i,0]=metrics1(z_train,np.mean(z_pred_train,axis=1,keepdims=True))[0]
    MSE[i,1]=metrics1(z_test,np.mean(z_pred_test,axis=1,keepdims=True))[0]
    r2[i,0]=metrics1(z_train,np.mean(z_pred_train,axis=1,keepdims=True))[1]
    r2[i,1]=metrics1(z_test,np.mean(z_pred_test,axis=1,keepdims=True))[1]
    error[i,0]=metrics2(z_train,z_pred_train)[0]
    error[i,1]=metrics2(z_test,z_pred_test)[0]
    bias[i,0]=metrics2(z_train,z_pred_train)[1]
    bias[i,1]=metrics2(z_test,z_pred_test)[1]
    variance[i,0]=metrics2(z_train,z_pred_train)[2]
    variance[i,1]=metrics2(z_test,z_pred_test)[2]

#%%
"""Plot results"""

plt.figure()
plt.plot(order,MSE[:,0],'r--',label='MSE train data')
plt.plot(order,MSE[:,1],'r-',label='MSE test data')
plt.xlabel('Model Complexity',fontweight='bold')
plt.ylabel('MSE',fontweight='bold')
plt.title('Noise SD=0.1, bootstrap=500, n=5000',fontweight='bold')
plt.grid()
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(order,bias[:,0],'b--',label='Bias train data')
plt.plot(order,bias[:,1],'b-',label='Bias test data')
plt.plot(order,variance[:,0],'g--',label='Var train data')
plt.plot(order,variance[:,1],'g-',label='Var test data')
plt.plot(order,error[:,0],'m--',label='Error train data')
plt.plot(order,error[:,1],'m-',label='Error test data')
plt.xlabel('Model Complexity',fontweight='bold')
plt.ylabel('Errors',fontweight='bold')
plt.title('Noise SD=0.1, bootstrap=500, n=5000',fontweight='bold')
plt.grid()
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(order,r2[:,0],'r--',label='$r^{2}$ train data')
plt.plot(order,r2[:,1],'r-',label='$r^{2}$ test data')
plt.xlabel('Model Complexity',fontweight='bold')
plt.ylabel('$r^{2}$',fontweight='bold')
plt.title('Noise SD=0.1, bootstrap=500, n=5000',fontweight='bold')
plt.grid()
plt.legend(loc='best')
plt.show()

plt.figure()
plt.plot(order,error[:,0],'m--',label='Total error train data')
plt.plot(order,error[:,1],'m-',label='Total error test data')
plt.plot(order,bias[:,0]+variance[:,0]+0.1**2,'k--',label='Sum of errors train data')
plt.plot(order,bias[:,1]+variance[:,1]+0.1**2,'k-',label='Sum of errors test data')
plt.xlabel('Model Complexity',fontweight='bold')
plt.ylabel('Errors',fontweight='bold')
plt.title('Noise SD=0.1, bootstrap=500, n=5000',fontweight='bold')
plt.grid()
plt.legend(loc='best')
plt.show()
