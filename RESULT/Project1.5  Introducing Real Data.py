# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 09:02:17 2021

@author: Alma Kurmanova and Yash Kumar
"""

#%%
""""Import all necessary packages"""

import numpy as np      
import pandas as pd
import seaborn as sns   
import matplotlib.pyplot as plt                                  
from sklearn.metrics import mean_squared_error as MSE            
from sklearn.metrics import r2_score as R2                       
from sklearn.model_selection import train_test_split             
from sklearn.preprocessing import StandardScaler                                              
from sklearn.datasets import load_boston 
from sklearn.preprocessing import PolynomialFeatures
import sklearn.linear_model as skl
from sklearn.model_selection import KFold                

#%%
"""Set the parameters"""

boston_data = load_boston() 
boston_df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston_df['MEDV'] = boston_data.target

#%%
"""Feature selection"""

corr_matrix = boston_df.corr().round(1)
sns.heatmap(data=corr_matrix, annot=True)

#%%
"""Create the dataset with chosen features"""

x=pd.DataFrame(np.c_[boston_df['RM'],boston_df['PTRATIO'],boston_df['INDUS']])
y=boston_df['MEDV']
x_arr=x.to_numpy()
y_arr=y.to_numpy()

#%%
"""Create the design matrix"""

order=3
poly = PolynomialFeatures(order)
X = poly.fit_transform(x_arr)    

#%%
"""Split the data into train and test and scale"""

X_train,X_test,y_train,y_test=train_test_split(X,y_arr,test_size=0.2,random_state=5) 
scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

#%%
"""Ridge Regression"""

lamda=np.logspace(-5,0,6)
MSE_train_Ridge=np.zeros((lamda.shape))
MSE_test_Ridge=np.zeros((lamda.shape))
r2_train_Ridge=np.zeros((lamda.shape))
r2_test_Ridge=np.zeros((lamda.shape))

for i in range(len(lamda)):
    clf=skl.Ridge(alpha=lamda[i],fit_intercept=True).fit(X_train_scaled,y_train)
    y_train_pred=clf.predict(X_train_scaled)
    y_test_pred=clf.predict(X_test_scaled)
    MSE_train_Ridge[i]=MSE(y_train,y_train_pred)
    MSE_test_Ridge[i]=MSE(y_test,y_test_pred)
    r2_train_Ridge[i]=R2(y_train,y_train_pred)
    r2_test_Ridge[i]=R2(y_test,y_test_pred)
    
#%%
"""OLS Regression"""

MSE_train_OLS=np.zeros((lamda.shape))
MSE_test_OLS=np.zeros((lamda.shape))
r2_train_OLS=np.zeros((lamda.shape))
r2_test_OLS=np.zeros((lamda.shape))
    
clf=skl.LinearRegression(fit_intercept=True).fit(X_train_scaled,y_train)
y_train_pred=clf.predict(X_train_scaled)
y_test_pred=clf.predict(X_test_scaled)
MSE_train_OLS[:]=MSE(y_train,y_train_pred)
MSE_test_OLS[:]=MSE(y_test,y_test_pred)
r2_train_OLS[:]=R2(y_train,y_train_pred)
r2_test_OLS[:]=R2(y_test,y_test_pred)

#%%
"""Plot results"""
plt.figure()
plt.plot(lamda,MSE_train_Ridge,'r-',label='Training MSE Ridge')
plt.plot(lamda,MSE_test_Ridge,'r--',label='Testing MSE Ridge')
plt.plot(lamda,MSE_train_OLS,'b-',label='Training MSE OLS')
plt.plot(lamda,MSE_test_OLS,'b--',label='Testing MSE OLS')
plt.xscale('log')
plt.legend(loc='best')
plt.xlabel('$\u03BB$')
plt.ylabel('MSE')
plt.grid()
plt.show()

plt.figure()
plt.plot(lamda,r2_train_Ridge,'r-',label='Training R2 Ridge')
plt.plot(lamda,r2_test_Ridge,'r--',label='Testing R2 Ridge')
plt.plot(lamda,r2_train_OLS,'b-',label='Training R2 OLS')
plt.plot(lamda,r2_test_OLS,'b--',label='Testing R2 OLS')
plt.xscale('log')
plt.legend(loc='best')
plt.xlabel('$\u03BB$')
plt.ylabel('$R^2$')
plt.grid()
plt.show()

#%%
"""K-fold CV for OLS with 3 features and order=3"""

k=np.linspace(5,10,6,dtype=int)
MSE_train_CV=np.zeros(k.shape)
R2_train_CV=np.zeros(k.shape)
MSE_test_CV=np.zeros(k.shape)
R2_test_CV=np.zeros(k.shape)


for i in range(len(k)):                                                       
    k1=k[i]
    kfold=KFold(n_splits=k1)
    for train,test in kfold.split(X):  
        j=0
        M_train=X[train]
        z_train=y[train]                 
        M_test=X[test]
        z_test=y[test]
        scaler=StandardScaler()                 
        scaler.fit(M_train)                     
        M_train=scaler.transform(M_train)      
        M_test=scaler.transform(M_test)         
        # z_train=np.reshape(z_train,(len(z_train),1))
        # z_test=np.reshape(z_test,(len(z_test),1))
        zpred_train=np.zeros((len(z_train),k1))  
        zpred_test=np.zeros((len(z_test),k1))       
        clf=skl.LinearRegression(fit_intercept=True).fit(M_train,z_train)    
        zpred_train[:,j]=clf.predict(M_train) 
        zpred_test[:,j]=clf.predict(M_test)    
        j+=1
    MSE_train_CV[i]=MSE(z_train,np.mean(zpred_train,axis=1,keepdims=True))    
    MSE_test_CV[i]=MSE(z_test,np.mean(zpred_test,axis=1,keepdims=True))       
    R2_train_CV[i]=R2(z_train,np.mean(zpred_train,axis=1,keepdims=True))      
    R2_test_CV[i]=R2(z_test,np.mean(zpred_test,axis=1,keepdims=True))        
    
"""Plot the results"""
plt.figure()
plt.plot(k,MSE_train_CV,'r-',label='Training MSE CV')
plt.plot(k,MSE_test_CV,'r--',label='Testing MSE CV')
plt.xscale('linear')
plt.legend(loc='best')
plt.xlabel('k')
plt.ylabel('MSE')
plt.grid()
plt.show()

plt.figure()
plt.plot(k,R2_train_CV,'r-',label='Training R2 CV')
plt.plot(k,R2_test_CV,'r--',label='Testing R2 CV')
plt.xscale('linear')
plt.legend(loc='best')
plt.xlabel('k')
plt.ylabel('$R^2$')
plt.grid()
plt.show()


    
    
