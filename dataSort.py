import numpy 			 as np

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