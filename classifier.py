#%matplotlib inline
import time

import numpy 				 as np
import numpy.matlib 		 as matlib

import matplotlib.pyplot 	as pyplot
import pandas 			 	as pd

from sklearn.decomposition 	import PCA

from keras 					import utils
from keras.models 			import Sequential
from keras.layers 			import Dense, Activation
from keras.optimizers 		import SGD
from keras.callbacks 		import EarlyStopping

import dataSort 		as dataSort

dataPath = ".\dataset\Dataset_spine.csv"
weightPath = ".\weights.txt"

data = pd.read_csv(dataPath)

#inputSize = data.shape[0]		# Number of entries in the dataset
K = 2							# Number of classes. 2 for binary classification
initNum = 250					# Number of random initializations

dataDim = data.shape[1]
data = data.drop(data.columns[dataDim-1], 1)	# Drop last columns, as it contains only comments
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

## Network architecture
neurons1 = 30
neurons2 = 0

resultsPath = ".\Results\\"  + time.strftime("%Y-%m-%d %Hh%Mm%S") + " N1 "+ str(neurons1) + " N2 "+ str(neurons2) + ".xls"

## Network hyperparameters
learningRate = 0.01
maxEpochs = 1000
batchSize = 64

eta 	  = np.empty(initNum)
metrics   = np.empty((initNum, 2))
numEpochs = np.empty(initNum)

for i in range(initNum):
	## Shuffle data
	x, y = dataSort.dataShuffle(x, y)

	# Split data
	trainSplit = 0.7
	#xFolds, yFolds = kFolds(x, y, trainSplit)
	x_train, y_train, x_test, y_test, x_val, y_val = dataSort.dataSplit(x, y, trainSplit)

	model = Sequential()

	#Input
	model.add(Dense(units=neurons1, input_dim=inputDim))
	model.add(Activation('tanh'))

	if neurons2 > 0:
		model.add(Dense(units=neurons2))
		model.add(Activation('tanh'))

	# Output
	model.add(Dense(units=K))
	model.add(Activation('softmax'))

	# Configure optimizer
	sgd = SGD(lr=learningRate)
	model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['categorical_accuracy'])

	# Configure callbacks
	earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1)

	# Train Network
	print("\nInit number: ", i+1)
	timerStart = time.time()

	hist = model.fit(x_train, y_train, epochs=maxEpochs, batch_size=batchSize, callbacks=[earlyStop] ,validation_data=(x_val, y_val), verbose=0)
	numEpochs[i-1] = len(hist.history['categorical_accuracy'])

	timerEnd = time.time()
	eta[i-1] = timerEnd-timerStart

	# Test trained model
	metrics[i-1,:] = model.evaluate(x_test, y_test, batch_size=batchSize)
	#y_pred = model.predict(x_test, batch_size=batchSize)

## Save to Excel file
results = pd.DataFrame({'Accuracy': metrics[:, 1], 'Loss': metrics[:, 0], 'Elapsed time': eta, 'Epochs': numEpochs, 'Acc Mean': np.mean(metrics[:, 1]), 'Acc Std': np.std(metrics[:, 1])})
results.to_excel(resultsPath, sheet_name='Results',  index=True)

# print("")
# print("")
# print(results.head())


## Information
# print(model.summary())
print('\n')
print("Epochs: ", np.mean(numEpochs))
print("Elapsed time: ", np.sum(eta))
print("Elapsed time per epoch: ", np.mean(eta)/np.sum(numEpochs))
print("Loss, Mean: ", np.mean(metrics[:, 0]))
print("Accuracy")
print("   Mean: ", np.mean(metrics[:, 1]))
print("   Std : ", np.std(metrics[:, 1]))

# # Show predictions
# print("----Predictions----")
# print("\nPrediction :", np.argmax(y_pred[0]))
# pyplot.imshow(np.reshape(x_test[0], (imgSizes)))