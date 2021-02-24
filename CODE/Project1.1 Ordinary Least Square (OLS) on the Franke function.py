# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 19:45:08 2021

@author: Alma Kurmanova and Yash Kumar
"""

"""Project 1 - Part (a) - Ordinary Least Squares Regression on Franke Function"""

#%%
"""Import all necessary packages"""

import numpy as np                                               #for numerical operations
import matplotlib.pyplot as plt                                  #for plotting operations
# from sklearn.metrics import mean_squared_error as MSE            #for calculating mean squared error
# from sklearn.metrics import r2_score as R2                       #for calculating r2
from sklearn.model_selection import train_test_split as splitter #for splitting data into train and test set 
from sklearn.preprocessing import StandardScaler                 #for scaling data as a normal distribution
from sklearn.preprocessing import MinMaxScaler                   #for scaing data between 0 and 1  
from matplotlib import cm                                        #for colormap
from mpl_toolkits.mplot3d import Axes3D                         
# import sklearn.linear_model as skl                               #for regression using scikit-learn    

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
plt.xlabel('X',fontweight='bold')
plt.ylabel('Y',fontweight='bold')
fig.colorbar(surf,shrink=0.5,aspect=5)       #colorbar properties
plt.show()

del x_plot,y_plot,z_plot,X,Y                 #clear memory

#%%
"""Define dataset for OLS fitting"""

x=np.linspace(0,1,1001).reshape(-1,1)         #define x array of 101 points between 0 and 1
y=np.linspace(0,1,1001).reshape(-1,1)         #define y array of 101 points between 0 and 1

z=FrankeFunction(x,y)+np.random.normal(0,0.1,x.shape) #calculate z array with noise

#%%
"""Define array for order of polynomial"""

order=np.arange(0,6,1)                      #define order array between 0 and 5

#%%
"""Define function for design matrix"""

def DesMatrix(degree,data1,data2):              #definition for function to make design matrix X
    X=np.ones((len(data1),1))                   #default design matrix with ones column 
    for i in range(degree+1):                   #loop to generate terms with powers of x and y
        if (i>0):
            for j in range(i+1):
                temp=(data1**j)*(data2**(i-j))
                X=np.hstack((X,temp))           #add terms to design matrix X
    return X

#%%
"""Define function for OLS regression"""

def OLS(X_train,X_test,y_train):            #definition for function to perform OLS fitting
    beta=np.linalg.pinv(X_train)@y_train    #calculate model parameters using pseudoinverse
    y_train=X_train@beta                    #predict training data
    y_test=X_test@beta                      #predict testing data 
    # clf=skl.LinearRegression(fit_intercept=True).fit(X_train,y_train)  #perform OLS regression using scikit-learn
    # y_train=clf.predict(X_train)                                       #predict training data 
    # y_test=clf.predict(X_test)                                         #predict testing data
    return y_train,y_test

#%%
"""Define function for parameter variances"""

def var_beta(X):                                  #definition for function to return parameter variances 
    U,S,VT=np.linalg.svd(X,full_matrices=True)    #perform SVD (X.T@X is singular, requires SVD)
    var=np.zeros((VT.shape[0],1))                
    for i in range(var.shape[0]):
        var[i]=np.sum((VT.T[i,:]/S[:])**2)        #calculate parameter variance using algorithm of Mandel
    return var

#%%
"""Define function to calculate MSE and r2 scores"""

def metrics(y_true,y_pred):                   #definition for function to return MSE and r2 scores
    mse=np.mean((y_true-y_pred)**2)
    # mse=MSE(y_true,y_pred)
    r2=1-np.sum((y_true-y_pred)**2)/np.sum((y_true-np.mean(y_true))**2)
    # r2=R2(y_true,y_pred)
    return mse,r2

#%%
"""Declare arrays to store MSE and r2-score and create empty list for parameter variances (no scaling)"""

MSE_unsc=np.zeros((order.shape[0],2))     #initialise arrays to store MSE of unscaled data
r2_unsc=np.zeros((order.shape[0],2))      #initialise arrays to store r2 of unscaled data
var_list_unsc=[]                          #initialise list to store parameter variances

#%%
"""Split data into training and testing data and perform regression (without scaling)"""
 
for i in range(len(order)):                      #loop over model complexity 
    X=DesMatrix(order[i],x,y)                    #construct design matrix
    X_train,X_test,z_train,z_test=splitter(X,z,test_size=0.2,random_state=12)   #split data into 80% train and 20% test 
    z_pred=OLS(X_train,X_test,z_train)           #perform OLS fitting
    var_list_unsc.append(var_beta(X_train))      #update parameter variance list 
    MSE_unsc[i,0]=metrics(z_train,z_pred[0])[0]  #calculate MSE of unscaled training data
    MSE_unsc[i,1]=metrics(z_test,z_pred[1])[0]   #calculate MSE of unscaled testing data
    r2_unsc[i,0]=metrics(z_train,z_pred[0])[1]   #calculate r2 of unscaled training data
    r2_unsc[i,1]=metrics(z_test,z_pred[1])[1]    #calculate r2 of unscaled testing data
    
#%%
"""Declare arrays to store MSE and r2-score and create empty list for parameter variances (standard scaling)"""

MSE_stsc=np.zeros((order.shape[0],2))         #initialise arrays to store MSE of std scaled data
r2_stsc=np.zeros((order.shape[0],2))          #initialise arrays to store r2 of std scaled data
var_list_stsc=[]                              #initialise list to store parameter variances

#%%
"""Split data into training and testing data and perform regression (with standard scaling)"""

for i in range(len(order)):                       #loop over model complexity 
    X=DesMatrix(order[i],x,y)                     #construct design matrix 
    X_train,X_test,z_train,z_test=splitter(X,z,test_size=0.2,random_state=12) #split data into 80% train and 20% test 
    scaler=StandardScaler()                       #initialise scaler
    scaler.fit(X_train)                           #evaluate mean and SD of model features to rescale as normal dist
    X_train_scaled=scaler.transform(X_train)      #normalise training design matrix to normal distribution
    X_test_scaled=scaler.transform(X_test)        #normalise testing design matrix to normal distribution
    z_pred=OLS(X_train_scaled,X_test_scaled,z_train)  #perform OLS fitting
    var_list_stsc.append(var_beta(X_train))       #update parameter variance list 
    MSE_stsc[i,0]=metrics(z_train,z_pred[0])[0]   #calculate MSE of unscaled training data
    MSE_stsc[i,1]=metrics(z_test,z_pred[1])[0]    #calculate MSE of unscaled testing data
    r2_stsc[i,0]=metrics(z_train,z_pred[0])[1]    #calculate r2 of unscaled training data
    r2_stsc[i,1]=metrics(z_test,z_pred[1])[1]     #calculate r2 of unscaled testing data
    
#%%
"""Declare arrays to store MSE and r2-score and create empty list for parameter variances (minmax scaling)"""

MSE_mnmxsc=np.zeros((order.shape[0],2))           #initialise arrays to store MSE of mnmx scaled data
r2_mnmxsc=np.zeros((order.shape[0],2))            #initialise arrays to store r2 of mnmx scaled data
var_list_mnmx=[]                                  #initialise list to store parameter variances

#%%
"""Split data into training and testing data and perform regression (with minmax scaling)"""

for i in range(len(order)):                    #loop over model complexity 
    X=DesMatrix(order[i],x,y)                  #construct design matrix                     
    X_train,X_test,z_train,z_test=splitter(X,z,test_size=0.2,random_state=12) #split data into 80% train and 20% test 
    scaler=MinMaxScaler()                      #initialise scaler
    scaler.fit(X_train)                        #evaluate mean and SD of model features to rescale as normal dist
    X_train_scaled=scaler.transform(X_train)   #normalise training design matrix to normal distribution
    X_test_scaled=scaler.transform(X_test)     #normalise testing design matrix to normal distribution
    z_pred=OLS(X_train_scaled,X_test_scaled,z_train)  #perform OLS fitting
    var_list_mnmx.append(var_beta(X_train))    #update parameter variance list 
    MSE_mnmxsc[i,0]=metrics(z_train,z_pred[0])[0]   #calculate MSE of unscaled training data
    MSE_mnmxsc[i,1]=metrics(z_test,z_pred[1])[0]    #calculate MSE of unscaled testing data
    r2_mnmxsc[i,0]=metrics(z_train,z_pred[0])[1]    #calculate r2 of unscaled training data
    r2_mnmxsc[i,1]=metrics(z_test,z_pred[1])[1]     #calculate r2 of unscaled testing data 
    
#%%
"""Plot results"""

plt.figure()
plt.plot(order,MSE_unsc[:,0],'r--',label='MSE train data (unscaled)')
plt.plot(order,MSE_unsc[:,1],'r-',label='MSE test data (unscaled)')
plt.plot(order,MSE_stsc[:,0],'b--',label='MSE train data (std scaled)')
plt.plot(order,MSE_stsc[:,1],'b-',label='MSE test data (std scaled)')
plt.plot(order,MSE_mnmxsc[:,0],'g--',label='MSE train data (mnmx. scaled)')
plt.plot(order,MSE_mnmxsc[:,1],'g-',label='MSE test data (mnmx. scaled)')
plt.xlabel('Model Complexity',fontweight='bold')
plt.ylabel('MSE',fontweight='bold')
plt.title('Model Analysis - Noise SD=0.1,n=1000 - MSE',fontweight='bold')
plt.grid()
plt.legend(loc='best',fontsize='x-small')
plt.show()

plt.figure()
plt.plot(order,r2_unsc[:,0],'r--',label='$r^{2}$ train data (unscaled)')
plt.plot(order,r2_unsc[:,1],'r-',label='$r^{2}$ test data (unscaled)')
plt.plot(order,r2_stsc[:,0],'b--',label='$r^{2}$ train data (std scaled)')
plt.plot(order,r2_stsc[:,1],'b-',label='$r^{2}$ test data (std scaled)')
plt.plot(order,r2_mnmxsc[:,0],'g--',label='$r^{2}$ train data (mnmx. scaled)')
plt.plot(order,r2_mnmxsc[:,1],'g-',label='$r^{2}$ test data (mnmx. scaled)')
plt.xlabel('Model Complexity',fontweight='bold')
plt.ylabel('$r^{2}$',fontweight='bold')
plt.title('Model Analysis - Noise SD=0.1,n=1000 - $r^{2}$',fontweight='bold')
plt.grid()
plt.legend(loc='best',fontsize='x-small')
plt.show()    

params=['$\u03B2_{0}$','$\u03B2_{1}$','$\u03B2_{2}$','$\u03B2_{3}$','$\u03B2_{4}$','$\u03B2_{5}$','$\u03B2_{6}$',
        '$\u03B2_{7}$','$\u03B2_{8}$','$\u03B2_{9}$','$\u03B2_{10}$','$\u03B2_{11}$','$\u03B2_{12}$','$\u03B2_{13}$',
        '$\u03B2_{14}$','$\u03B2_{15}$','$\u03B2_{16}$','$\u03B2_{17}$','$\u03B2_{18}$','$\u03B2_{19}$','$\u03B2_{20}$',
        '$\u03B2_{21}$']

plt.figure()
plt.plot(params[0:len(var_list_unsc[2])],np.sqrt(var_list_unsc[2]),'r--',label='Deg=2, unscaled')
plt.plot(params[0:len(var_list_unsc[2])],np.sqrt(var_list_stsc[2]),'r-',label='Deg=2, std scaled)')
plt.plot(params[0:len(var_list_unsc[2])],np.sqrt(var_list_mnmx[2]),'r-o',label='Deg=2, mnmx. scaled')
plt.plot(params[0:len(var_list_unsc[5])],np.sqrt(var_list_unsc[5]),'b--',label='Deg=5, unscaled')
plt.plot(params[0:len(var_list_unsc[5])],np.sqrt(var_list_stsc[5]),'b-',label='Deg=5, std scaled)')
plt.plot(params[0:len(var_list_unsc[5])],np.sqrt(var_list_mnmx[5]),'b-o',label='Deg=5, mnmx. scaled')
plt.xlabel('\u03B2')
plt.ylabel('$\u03C3$')
plt.yscale('log')
plt.grid()
plt.legend(loc='best',fontsize='small')
plt.show()


    


      


