#%matplotlib inline
import time

import numpy 				 as np
import numpy.matlib 		 as matlib

import matplotlib.pyplot 	as pyplot
import pandas 			 	as pd

from sklearn.decomposition 	import PCA

from keras 					import utils
# from keras.models 			import Sequential
# from keras.layers 			import Dense, Activation
# from keras.optimizers 		import SGD
# from keras.callbacks 			import EarlyStopping

import dataSort 		as dataSort

dataPath = ".\dataset\Dataset_spine.csv"
weightPath = ".\weights.txt"

data = pd.read_csv(dataPath)

#inputSize = data.shape[0]		# Number of entries in the dataset
K = 2							# Number of classes. 2 for binary classification
initNum = 250					# Number of random initializations

dataDim = data.shape[1]
data = data.drop(data.columns[dataDim-1], 1)	# Drop last column, as it contains only comments
dataDim = data.shape[1]
 
# print(data.shape)
# print(data.head())

## Shuffle data
#data = data.sample(frac=1).reset_index(drop=True)

x = data.iloc[:, :dataDim-1].as_matrix()
y = data.iloc[:, dataDim-1].as_matrix()

inputDim = x.shape[1]

## Unwrap labels
y = np.where(y == "Abnormal", 1, 0)	# where(condition, True value, False value)
y = utils.to_categorical(y, K)		# y[:, 0] -> Positive Class
									# y[:, 1] -> Negative Class

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
# pcaPercentage = 0.8
# pcaX = PCA(n_components=pcaPercentage)
# x = pcaX.fit_transform(x)

inputDim = x.shape[1]			# New shape after compression

# print("")
# print("--Depois de PCA--")
# print("X shape: ", x.shape)
# print(x[:10])

## Intruder Removal
# intruders = [75, 95, 115, 201, 224] 	# x > 3 std
intruders = [115] 						# x > 5 std
# intruders = -1						# to find intruders
x, y = dataSort.intruderRemoval(x, y, 5, intruders)

print("")
print("--Depois de IR--")
print("Y shape: ", y.shape)
print("X shape: ", x.shape)
print(x[:10])

print("\nClass populations: ", np.sum(y, 0))

folds=5
for index in range(folds):
	x_train, y_train, x_test, y_test, x_val, y_val = dataSort.kFolds(x, y, folds, index)

	print("x train shape: ", x_train.shape)
	print("y train  shape: ", y_train.shape)

	print("x  test shape: ", x_test.shape)
	print("y  test shape: ", y_test.shape)

	print("x val shape: ", x_val.shape)
	print("y val shape: ", y_val.shape)

	print("\nx train:\n", x_train[:10])
	print("\ny train:\n", y_train[:10])