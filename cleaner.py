#%matplotlib inline
import time

import numpy 			 as np
import numpy.matlib 	 as matlib
import matplotlib.pyplot as pyplot
import pandas 			 as pd

from sklearn.decomposition 	import PCA
from keras 					import utils

import dataSort 		as dataSort

dataPath = ".\dataset\Dataset_spine.csv"
weightPath = ".\weights.txt"

data = pd.read_csv(dataPath)

#inputSize = data.shape[0]		# Number of entries in the dataset
K = 2							# Number of classes. 2 for binary classification

dataDim = data.shape[1]
data = data.drop(data.columns[dataDim-1], 1)	# Drop last columns, as it contains only comments
dataDim = data.shape[1]
 
# print(data.shape)
# print(data.head())

x = data.iloc[:, :dataDim-1].as_matrix()
y = data.iloc[:, dataDim-1].as_matrix()

inputDim = x.shape[1]

## Unwrap labels
y = np.where(y == "Abnormal", 1, 0)	# where(condition, True value, False value)
y = utils.to_categorical(y, K)

print("")
print("--Original--")
print("X shape: ", x.shape)
print(x[:10])

## Input Normalization and scaling
xMeans = np.mean(x, keepdims=True, dtype=np.float64)
xStds  = np.std(x, keepdims=True, dtype=np.float64)
x = (x - xMeans)/xStds

print("")
print("--Depois de norm--")
print("X shape: ", x.shape)
print(x[:10])

## Apply PCA for dimensionality reduction
pcaPercentage = 0.8
pcaX = PCA(n_components=pcaPercentage)
x = pcaX.fit_transform(x)

inputDim = x.shape[1]			# New shape after compression

print("")
print("--Depois de PCA--")
print("X shape: ", x.shape)
print(x[:10])

x, y = dataSort.intruderRemoval(x, y, 3)

print("")
print("--Depois de IR--")
print("Y shape: ", y.shape)
print("X shape: ", x.shape)
print(x[:10])


## Shuffle data
data = data.sample(frac=1).reset_index(drop=True)