import numpy 			 as np

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
	# TODO: FIX THIS
	# for each dimension of x, take mean and std
	xMeans = np.mean(x, axis=1, keepdims=True, dtype=np.float64)
	xStds  = np.std(x, axis=1, keepdims=True, dtype=np.float64)

	print("xMeans shape: ", xMeans.shape)
	print("xStds shape: ", xStds.shape)

	mean = np.mean(xMeans, dtype=np.float64)
	std  = np.mean(xStds, dtype=np.float64)

	index = np.where(x <= limit*xStds, 1, 0)
	xNew = x[index == 1]
	# yNew = y[index == 1]

	return xNew#, yNew

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