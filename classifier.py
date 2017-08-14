#%matplotlib inline
import time

import numpy 			 as np
import numpy.matlib 	 as matlib
import matplotlib.pyplot as pyplot
import pandas 			 as pd

# from keras 				import utils
# from keras.models 		import Sequential
# from keras.layers 		import Dense, Activation
# from keras.optimizers 	import SGD
# from keras.callbacks 	import EarlyStopping

dataPath = ".\dataset\Dataset_spine.csv"

data = pd.read_csv(dataPath)

inputSize = data.shape[0]

dim = data.shape[1]
data = data.drop(data.columns[dim-1], 1)
dim = data.shape[1]

print(data.shape)
print(data.head())

# Shuffle data

data = data.sample(frac=1).reset_index(drop=True)

x = data.iloc[:, :dim-1]
y = data.iloc[:, dim-1]

print("")
print("X shape: ", x.shape)
print(x.head())

print("")
print("Y shape: ", y.shape)
print(y.head())