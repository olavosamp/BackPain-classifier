import numpy as np
import pandas 			 as pd
import dataSort 		as dataSort


# import time

# neurons1 = 10
# resultsPath = ".\Results\\"  + time.strftime("%Y-%m-%d %Hh%Mm%S") + " N "+ str(neurons1) + ".xls"
# print(resultsPath)


# x = np.arange(10)

# index = np.where(x <= 5, 1, 0)
# xNew = x[index == 1]

# print("X: ", x)
# print("")
# print("xNew: ", xNew)

dataPath = ".\dataset\Dataset_spine.csv"
weightPath = ".\weights.txt"
pcaPercentage = 0.8

data = pd.read_csv(dataPath)

inputSize = data.shape[0]		# Number of entries in the dataset
K = 2							# Number of classes. Always equals 2 for binary classification

dataDim = data.shape[1]
data = data.drop(data.columns[dataDim-1], 1)	# Drop last columns, as it contains only comments
dataDim = data.shape[1]
 
# print(data.shape)
# print(data.head())

## Shuffle data
# data = data.sample(frac=1).reset_index(drop=True)

x = data.iloc[:, :dataDim-1].as_matrix()
y = data.iloc[:, dataDim-1].as_matrix()

## Unwrap labels
y = np.where(y == "Abnormal", 1, 0)	# where(condition, True value, False value)
# y = utils.to_categorical(y, K)

dim = x.shape[1]

# print("")
# print("X shape: ", x.shape)
# print(x[:5])

## Input Normalization and scaling
xMeans = np.mean(x, keepdims=True, dtype=np.float64)
xStds  = np.std(x, keepdims=True, dtype=np.float64)
x = (x - xMeans)/xStds

# x = dataSort.intruderRemoval(x, y, 2)
xMeans = np.mean(x, axis=0, keepdims=True, dtype=np.float64)
xStds  = np.std(x, axis=0, keepdims=True, dtype=np.float64)

# print("")
# print("X means shape: ", xMeans.shape)
# print(xMeans[:10])
# print("X stds shape: ", xStds.shape)
# print(xStds[:10])

# for i in dim
# 	if x[:, i] > 2*xStds[i]
# 		x = np.delete(x, )

# print("")
# print("---After IR---")
# print("X shape: ", x.shape)
# print(x[:10])

# A = np.arange(10)
# B = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])

xTest = x[:10,:]
print("")
print("xTest shape: ", xTest.shape)
print(xTest)
xGauss = dataSort.gaussian(xTest, xMeans, xStds)
print("")
print("xGauss shape: ", xGauss.shape)
print(xGauss)

print("")
print("xStds shape: ", xStds.shape)
print(5*xStds)

print("Mask")
mask = np.absolute(xGauss) > 5*xStds
print(mask.shape)
print(mask)

# print("Sum y", np.sum(y))