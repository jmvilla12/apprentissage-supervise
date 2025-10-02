#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:15:50 2019

@author: garnier
"""
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('data/house_price.csv')

# data information
print('----------')
print("Data information")
print('----------')
data.info()
print('----------')
print("First data of the file")
print('----------')
print(data.head(10))

# data analysis
print('----------')
print("Statistical properties")
print('----------')
print(data.describe())
print('----------')
print("Table size")
print('----------')
nb_val, nb_col = data.shape
print(data.shape)

# store the output data 
# in a Dataframe (pandas series)
#y_data = data['SalePrice']
# in an array (numpy) with a shape (n_samples, ) or or (n_samples, 1)
y_data = np.array(data['SalePrice'])

# store and preprocess the input data (features)

# drop the Id and SalePrice columns 
data.drop(['Id', 'SalePrice'], axis = 1, inplace = True)

# drop columns that have a lot of missing or NaN values
cols_to_drop = ['MiscFeature', 'PoolQC', 'Fence', 'FireplaceQu', 'Alley']
data.drop(cols_to_drop, axis = 1, inplace = True)

# fill columns that have a few NaN values
cols_with_nan = ['LotFrontage', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
 'BsmtFinType2', 'Electrical', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']

for col in cols_with_nan:
    if data[col].dtype == 'object':
        data.fillna(data[col].value_counts().index[0], inplace = True)
    else:
        data.fillna(data[col].mean(), inplace = True)
        
# convert columns with object datatypes into numbers by using Label Encoding
object_cols = [col for col in data.columns if data[col].dtype == 'object']

for col in object_cols:
    data[col] = data[col].astype('str')
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# store the preprocessed input data  
# in a Dataframe (pandas)
#X_data = data[:]
# in an array (numpy) with a shape (n_samples, n_features)
X_data = np.array(data)

np.save('data/data_processed_X.npy',X_data)
np.save('data/data_processed_y.npy',y_data)


# data information after transformation
print('----------')
print("Data after transformation")
print('----------')
print("Data information")
print('----------')
data.info()
print('----------')
print("First data of the file")
print('----------')
print(data.head(10))

# data analysis after transformation
print('----------')
print("Statistical properties")
print('----------')
print(data.describe())
print('----------')
print("Table size")
print('----------')
nb_val, nb_col = data.shape
print(data.shape)



