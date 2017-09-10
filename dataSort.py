import numpy 			 as np
import math

## Shuffle / permute dataset
def dataShuffle(x, y):
	inputSize = x.shape[0]

	index = np.random.permutation(inputSize)
	newX = x[index]
	newY = y[index]

	return newX, newY

## Split the dataset in Training, Test and Validation
def dataSplit(x, y, trainSplit, testSplit=0, valSplit=0):
	m = x.shape[0]				# Dataset size

	if testSplit == 0:
		testSplit  = (1-trainSplit)/2

	if valSplit == 0:
		valSplit   = testSplit

	trainIndex = np.floor(m*trainSplit).astype(int)
	testIndex  = np.floor(m*testSplit).astype(int) + trainIndex

	x_train = x[:trainIndex]
	y_train = y[:trainIndex]

	x_test  = x[trainIndex:testIndex]
	y_test  = y[trainIndex:testIndex]

	x_val   = x[testIndex:]
	y_val   = y[testIndex:]

	return x_train, y_train, x_test, y_test, x_val, y_val

## Split-balance
def splitBalance(xNeg, yNeg, xPos, yPos, trainSplit):
	x_trainNeg, y_trainNeg, x_testNeg, y_testNeg, x_valNeg, y_valNeg = dataSplit(xNeg, yNeg, trainSplit)
	x_trainPos, y_trainPos, x_testPos, y_testPos, x_valPos, y_valPos = dataSplit(xPos, yPos, trainSplit)

	# Double positive class
	x_trainPos = np.concatenate((x_trainPos, x_trainPos))
	y_trainPos = np.concatenate((y_trainPos, y_trainPos))

	x_train = np.concatenate((x_trainNeg, x_trainPos))
	y_train = np.concatenate((y_trainNeg, y_trainPos))

	x_test = np.concatenate((x_testNeg, x_testPos))
	y_test = np.concatenate((y_testNeg, y_testPos))

	x_val = np.concatenate((x_valNeg, x_valPos))
	y_val = np.concatenate((y_valNeg, y_valPos))

	return x_train, y_train, x_test, y_test, x_val, y_val

# Split dataset in two classes
def popSplit(x, y):
	iNeg = np.where(y[:, 1] == 1, 1, 0)
	iPos = np.where(y[:, 0] == 1, 1, 0)

	yNeg = y[iNeg == 1]
	yPos = y[iPos == 1]

	xNeg = x[iNeg == 1]
	xPos = x[iPos == 1]

	# Double positive class
	# xPos = np.concatenate((xPos, xPos))
	# yPos = np.concatenate((yPos, yPos))

	return xNeg, yNeg, xPos, yPos

## Intruder removal
def intruderRemoval(x, y, limit, intruders=-1):
	# Set intruders = -1 to find the intruders
	# Set intruders to an array to remove pre-selected elements
	if intruders == -1:
		xStds  = np.std(x, axis=0, keepdims=True, dtype=np.float64)
		mask = np.absolute(x) > np.multiply(xStds, limit)

		intruders = np.where(np.sum(mask, 1))

	print("intruder indexes: \n", intruders)

	xNew = np.delete(x, intruders, 0)
	yNew = np.delete(y, intruders, 0)

	return xNew, yNew

## K-Folds training
def kFolds(x, y, folds=5, index=1):
	index = index%folds

	m   = x.shape[0]				# Dataset size
	dim = x.shape[1]				# Input dimension

	trainFolds = int(0.6*folds)		# Training: 	3 folds
	testFolds = int(0.2*folds)		# Test and val: 1 fold/ea

	# The flooring causes trailing elements to be discarded
	foldSize = np.floor(m/folds).astype(int) 

	xFolds = np.empty((folds, foldSize, dim))
	yFolds = np.empty((folds, foldSize, 2))

	# Create fold array
	for i in range(folds-1):
		xFolds[i] = x[i*foldSize:(i+1)*foldSize]
		yFolds[i] = y[i*foldSize:(i+1)*foldSize]

	# Shift the folds
	xFolds = np.roll(xFolds, index, 0)
	yFolds = np.roll(yFolds, index, 0)

	# Assign folds to each set
	x_train = np.reshape(xFolds[:trainFolds], (trainFolds*foldSize, dim))
	y_train = np.reshape(yFolds[:trainFolds], (trainFolds*foldSize, 2))

	x_test  = np.reshape(xFolds[trainFolds:trainFolds+testFolds], (testFolds*foldSize, dim))
	y_test  = np.reshape(yFolds[trainFolds:trainFolds+testFolds], (testFolds*foldSize, 2))

	x_val   = np.reshape(xFolds[trainFolds+testFolds:], (testFolds*foldSize, dim))
	y_val   = np.reshape(yFolds[trainFolds+testFolds:], (testFolds*foldSize, 2))

	return x_train, y_train, x_test, y_test, x_val, y_val

# Compute Gaussian PDF for given x
def gaussian(x, mu, sig):
    return 1./(math.sqrt(2.*math.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)