# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:28:00 2020

@author: stan
"""

#import numpy as np
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectPercentile
#pd.test()
#from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import DecisionTreeRegressor
#import matplotlib.pyplot as plt
#from sklearn.linear_model import Ridge

bd1 = pd.read_csv('train_data_200k.csv', delimiter = ',')
bd0 = pd.read_csv('test_data_100k.csv' ,delimiter = ',')
bd1=pd.DataFrame(bd1)
bd2=pd.DataFrame(bd0)



bd1=bd1.drop([0,1,2,3,4,5,6,7,8])
bd1=bd1.reset_index(drop=True)
bd2=bd2.reset_index(drop=True)

bd1=bd1.drop(bd1.columns[0], axis='columns')
bd2=bd2.drop(bd2.columns[0], axis='columns')

bd1=bd1.fillna(bd1.median())
bd2=bd2.fillna(bd1.median())

X=bd1.drop(['target1','target2','target3','target4'], axis='columns')
y=bd1[['target1','target2','target3','target4']]
#y0=bd1['target4']

#X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)

scaler = RobustScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_out=scaler.transform(bd2)


bd1['target']=(bd1['target1']**2+bd1['target2']**2+bd1['target3']**2+bd1['target4']**2)
y0=bd1['target']
X_train, X_test, y0_train, y_test = train_test_split(X, y0,random_state=1)


select = SelectPercentile(percentile=12.8)
select.fit(X_train_scaled, y0_train)
X_train_selected = select.transform(X_train_scaled)
print("shape of X_train_selected: {}".format(X_train_selected.shape))
mask = select.get_support()
print(mask)

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
#model=DecisionTreeRegressor(max_depth=20).fit(X_train_scaled, y_train)

#model=RandomForestRegressor(max_depth=20).fit(X_train_scaled, y_train) #n_estimators=5, random_state=2
model=RandomForestRegressor(n_estimators=10, max_features=30, random_state=2).fit(X_train_scaled, y_train)

print("accuracy on train set: {:.4f}".format(model.score(X_train_scaled, y_train)))
print("accuracy on test set: {:.4f}".format(model.score(X_test_scaled, y_test)))



y_out=model.predict(X_out)

y_out=pd.DataFrame(y_out)
y_out.columns=['target1','target2','target3','target4']

#Predict=bd0.merge(y_out, left_on='index')

y_out.to_csv('predict_100k.csv')












print('ura!!')