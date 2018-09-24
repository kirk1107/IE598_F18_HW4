#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 19:48:25 2018

@author: kirktsui
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error



target_url = ("https://raw.githubusercontent.com/rasbt/python-machine-"
              "learning-book-2nd-edition/master/code/ch10/housing.data.txt")

Housing = DataFrame(pd.read_csv(target_url,header=None, sep='\s+'))


Housing.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS',
                   'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

#print head and tail of data frame
print(Housing.head())
print(Housing.tail())
#print summary of data frame
summary = Housing.describe()
print(summary) 
 

sns.pairplot(Housing)
plt.tight_layout()
plt.show()


#HEATMAP
cm = np.corrcoef(Housing.values.T)
sns.set(font_scale=1.2)
hm = sns.heatmap(cm, 
                 cbar=True,
                 annot=True, 
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 7},
                 yticklabels=Housing.columns,
                 xticklabels=Housing.columns)
plt.show()


#BOXPLOT
array = Housing.iloc[:,0:506].values
plt.boxplot(array)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14],Housing.columns, rotation = 55)
plt.ylabel(("Quartile Ranges")) 
plt.show()


X = Housing.iloc[:,0:13]
y = Housing.iloc[:,13]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
print( X_train.shape, y_train.shape)


#STANDARDIZE
sc_x_tr = StandardScaler()
sc_y_tr = StandardScaler()
X_train_std = sc_x_tr.fit_transform(X_train)
y_train_std = sc_y_tr.fit_transform(y_train[:, np.newaxis]).flatten()

sc_x_te = StandardScaler()
sc_y_te = StandardScaler()
X_test_std = sc_x_te.fit_transform(X_test)
y_test_std = sc_y_te.fit_transform(y_test[:, np.newaxis]).flatten()


#LINEAR
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train_std, y_train_std)
print('Intercept:\t%.3f' % lr.intercept_)
#print('%.3f' % lr.intercept_)
for i in range (13):
    print('Slope #%.0f:\t%.3f' %(i+1, lr.coef_[i]))
#    print('%.3f' %(lr.coef_[i]))
y_train_pred = lr.predict(X_train_std)
y_test_pred = lr.predict(X_test_std)


#RESIDUAL PLOT
plt.scatter(y_train_pred,  y_train_pred - y_train_std,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
plt.scatter(y_test_pred,  y_test_pred - y_test_std,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')

plt.xlabel('Predicted values (standardized)')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-4, xmax=3, color='black', lw=2)
plt.show()

print('MSE train: %.3f, test: %.3f' % (
      mean_squared_error(y_train_std, y_train_pred),
      mean_squared_error(y_test_std, y_test_pred)))

from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f' % 
      (r2_score(y_train_std, y_train_pred),
       r2_score(y_test_std, y_test_pred)))


#RIDGE REGRESSION
from sklearn.linear_model import Ridge

alpha_space = np.logspace(-4, 0.5, 5)
R2_test = []
R2_train = []
alpha_serie = []
ridge = Ridge(normalize = True)

for alpha in alpha_space:
    ridge.alpha = alpha
    ridge.fit(X_train_std, y_train_std)
    y_train_pred = ridge.predict(X_train_std)
    y_test_pred = ridge.predict(X_test_std)
    print('alpha = %.4f'%alpha)
    print('\tIntercept:\t%.3f' % ridge.intercept_)
#    print('%.3f' % ridge.intercept_)
    for i in range (13):
      print('\tSlope #%.0f:\t%.3f' %(i+1, ridge.coef_[i])) 
#      print('%.3f' %( ridge.coef_[i])) 
    
    
 
    print('\tMSE train: %.3f, test: %.3f \tR^2 train: %.3f, test: %.3f' % 
      (mean_squared_error(y_train_std, y_train_pred),
       mean_squared_error(y_test_std, y_test_pred), 
       r2_score(y_train_std, y_train_pred),
       r2_score(y_test_std, y_test_pred)))
    
    alpha_serie.append(alpha)
    R2_test.append(r2_score(y_test_std, y_test_pred))
    R2_train.append(r2_score(y_train_std, y_train_pred))
    
    #RESIDUAL PLOT
    plt.title("Residual when alpha = %f"%alpha)
    plt.scatter(y_train_pred,  y_train_pred - y_train_std,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test_std,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values (standardized)')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-4, xmax=3, color='black', lw=2)
    plt.show()

plt.title("R-Squared by RIDGE with Different Alphas")
plt.plot(alpha_serie, R2_test)
plt.plot(alpha_serie, R2_train)
plt.xlabel("Alpha")
plt.ylabel("R^2")
plt.legend(labels = ['test','train'])
plt.xscale('log')
plt.show()



#LASSO

from sklearn.linear_model import Lasso

alpha_space = np.logspace(-4, 0.5, 5)
R2_test = []
R2_train = []
alpha_serie = []
lasso = Lasso(normalize = True)

for alpha in alpha_space:
    lasso.alpha = alpha
    lasso.fit(X_train_std, y_train_std)
    y_train_pred = lasso.predict(X_train_std)
    y_test_pred = lasso.predict(X_test_std)
    
    print('alpha = %.4f'%alpha)
    print('\tIntercept:\t%.3f' % lasso.intercept_)
#    print('%.3f' % lasso.intercept_)
    for i in range (13):
#        print('%.3f' %(lasso.coef_[i])) 
        print('\tSlope #%.0f:\t%.3f' %(i+1, lasso.coef_[i]))  
    
    print('\tMSE train: %.3f, test: %.3f \tR^2 train: %.3f, test: %.3f' % 
      (mean_squared_error(y_train_std, y_train_pred),
       mean_squared_error(y_test_std, y_test_pred), 
       r2_score(y_train_std, y_train_pred),
       r2_score(y_test_std, y_test_pred)))
    alpha_serie.append(alpha)
    R2_test.append(r2_score(y_test_std, y_test_pred))
    R2_train.append(r2_score(y_train_std, y_train_pred))
    
    plt.title("Residual when alpha = %f"%alpha)
    plt.scatter(y_train_pred,  y_train_pred - y_train_std,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test_std,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values (standardized)')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-4, xmax=3, color='black', lw=2)
    plt.show()
 
plt.title("R-Squared by LASSO with Different Alphas")
plt.plot(alpha_serie, R2_test)
plt.plot(alpha_serie, R2_train)
plt.xlabel("Alpha")
plt.ylabel("R^2")
plt.legend(labels = ['test','train'])
plt.xscale('log')
plt.show()


#ELASTICNET
from sklearn.linear_model import ElasticNet

l1_ratio_space = np.logspace(-2, 0.5, 5)
#l1_ratio_space = [0,1, 0.5, 1, 2, 3,]
R2_test = []
R2_train = []
l1_ratio_serie = []
eln = ElasticNet(normalize = True, alpha = 0.001)

for l1_ratio in l1_ratio_space:
    eln.l1_ratio = l1_ratio
    eln.fit(X_train_std, y_train_std)
    y_train_pred = eln.predict(X_train_std)
    y_test_pred = eln.predict(X_test_std)
    
    print('l1_ratio = %.4f'%l1_ratio)
    print('\tIntercept:\t%.3f' % eln.intercept_)
#    print('%.3f' % eln.intercept_)
    for i in range (13):
        print('\tSlope #%.0f:\t%.3f' %(i+1, eln.coef_[i])) 
#        print('%.3f' %(eln.coef_[i]))  
    
    print('\tMSE train: %.3f, test: %.3f \tR^2 train: %.3f, test: %.3f' % 
      (mean_squared_error(y_train_std, y_train_pred),
       mean_squared_error(y_test_std, y_test_pred), 
       r2_score(y_train_std, y_train_pred),
       r2_score(y_test_std, y_test_pred)))
    l1_ratio_serie.append(l1_ratio)
    R2_test.append(r2_score(y_test_std, y_test_pred))
    R2_train.append(r2_score(y_train_std, y_train_pred))
    
    
    plt.title("Residual when l1_ratio = %f"%l1_ratio)
    plt.scatter(y_train_pred,  y_train_pred - y_train_std,
            c='steelblue', marker='o', edgecolor='white',
            label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test_std,
            c='limegreen', marker='s', edgecolor='white',
            label='Test data')
    plt.xlabel('Predicted values (standardized)')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-2, xmax=1.5, color='black', lw=2)
    plt.show()
  
plt.title("R-Squared by ElasticNet with Different l1_ratios")
plt.plot(l1_ratio_serie, R2_test)
plt.plot(l1_ratio_serie, R2_train)
plt.xlabel("l1_ratio")
plt.ylabel("R^2")
plt.legend(labels = ['test','train'])
plt.xscale('log')
plt.show()


print("My name is Jianhao Cui")
print("My NetID is: jianhao3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
