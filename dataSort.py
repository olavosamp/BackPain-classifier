import numpy 			 as np
import math

# Shuffle / permute dataset
def dataShuffle(x, y):
	inputSize = x.shape[0]

	index = np.random.permutation(inputSize)
	newX = x[index]
	newY = y[index]

	return newX, newY

# Split the dataset in Training, Test and Validation
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

# Intruder removal
def intruderRemoval(x, y, limit):
	xStds  = np.std(x, axis=0, keepdims=True, dtype=np.float64)
	
	print("xStds shape: ", xStds.shape)

	mask = np.absolute(x) > np.multiply(xStds, limit)
	
	print("Mask")
	#mask = np.absolute(xTest) > xNewStds
	print(mask.shape)
	print(np.sum(mask, 1).shape)

	intruders = np.where(np.sum(mask, 1))
	print("intruder indexes: \n", intruders)

	xNew = np.delete(x, intruders, 0)
	yNew = np.delete(y, intruders, 0)

	return xNew, yNew

# Perform K-Folds training
def kFolds(x, y, trainSplit):
	m = x.shape[0]				# Dataset size
	folds = 20					# Test and validation sets will be determined from the training set
								# trainSplit must be a multiple of 5

	testSplit = (1-trainSplit)/2
	valSplit = testSplit

	foldSize = np.floor(m/folds).astype(int)

	if (folds*foldSize) != m:
		print("Error: Folds don't fit the data")
		return 1

	if (folds*foldSize) == m:
		xNewShape = (foldSize, x.shape[1], folds)
		yNewShape = (foldSize, y.shape[1], folds)

		xFolds = np.reshape(x, xNewShape)
		yFolds = np.reshape(y, yNewShape)

	return xFolds, yFolds

def gaussian(x, mu, sig):
    return 1./(math.sqrt(2.*math.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)