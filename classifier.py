#%matplotlib inline
import time

import numpy 			 as np
import numpy.matlib 	 as matlib
import matplotlib.pyplot as pyplot
import pandas 			 as pd

from keras 				import utils
from keras.models 		import Sequential
from keras.layers 		import Dense, Activation
from keras.optimizers 	import SGD
from keras.callbacks 	import EarlyStopping

from dataSort			import dataSplit

dataPath = ".\dataset\Dataset_spine.csv"

data = pd.read_csv(dataPath)

initNum = 2
inputSize = data.shape[0]
K = 2

dataDim = data.shape[1]
data = data.drop(data.columns[dataDim-1], 1)
dataDim = data.shape[1]

print(data.shape)
print(data.head())

# Shuffle data
# data = data.sample(frac=1).reset_index(drop=True)

x = data.iloc[:, :dataDim-1].as_matrix()
y = data.iloc[:, dataDim-1].as_matrix()

# Unwrap labels
y = np.where(y == "Abnormal", 1, 0)
y = utils.to_categorical(y, K)

inputDim = x.shape[1]

# Input Normalization and scaling
xMeans = np.mean(x, keepdims=True, dtype=np.float64)
xStds  = np.std(x, keepdims=True, dtype=np.float64)
x = (x - xMeans)/xStds

# print("")
# print("X shape: ", x.shape)
# print(x[:5])

# print("")
# print("Y shape: ", y.shape)
# print(y)

metrics = np.empty((initNum, 2))
eta = np.empty(initNum)
numEpochs = np.empty(initNum)

for i in range(initNum):
	# Shuffle dataset
	index = np.random.permutation(inputSize)
	x = x[index]
	y = y[index]

	# Split data
	trainSplit = 0.7
	x_train, y_train, x_test, y_test, x_val, y_val = dataSplit(x, y, trainSplit)

	model = Sequential()

	# Network architecture
	neurons1 = 5
	# neurons2 = 50

	# Network hyperparameters
	learningRate = 0.01
	maxEpochs = 1000
	batchSize = 8

	#Input
	model.add(Dense(units=neurons1, input_dim=inputDim))
	model.add(Activation('tanh'))

	# model.add(Dense(units=neurons2))
	# model.add(Activation('tanh'))

	# Output
	model.add(Dense(units=K))
	model.add(Activation('softmax'))

	# Configure optimizer
	sgd = SGD(lr=learningRate)
	model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

	# Configure callbacks
	earlyStop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1)

	# Train Network
	timerStart = time.time()

	hist = model.fit(x_train, y_train, epochs=maxEpochs, batch_size=batchSize, callbacks=[earlyStop] ,validation_data=(x_val, y_val), verbose=0)
	numEpochs[i-1] = len(hist.history['acc'])

	timerEnd = time.time()
	eta[i-1] = timerEnd-timerStart

	# Test trained model
	metrics[i-1,:] = model.evaluate(x_test, y_test, batch_size=batchSize)
	#y_pred = model.predict(x_test, batch_size=batchSize)


# Information
print('\n')
print(metrics)
print(eta)
#print(model.summary())

print('\n')
print("Epochs: ", np.mean(numEpochs))
print("Elapsed time: ", np.sum(eta))
print("Elapsed time per epoch: ", np.mean(eta)/np.sum(numEpochs))
print("Loss: ", np.mean(metrics[:, 0]))
print("Accuracy: ", np.mean(metrics[:, 1]))

# # Show predictions
# print("----Predictions----")
# print("\nPrediction :", np.argmax(y_pred[0]))
# pyplot.imshow(np.reshape(x_test[0], (imgSizes)))