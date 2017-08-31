import numpy as np

# import time

# neurons1 = 10
# resultsPath = ".\Results\\"  + time.strftime("%Y-%m-%d %Hh%Mm%S") + " N "+ str(neurons1) + ".xls"
# print(resultsPath)


x = np.arange(10)

index = np.where(x <= 5, 1, 0)
xNew = x[index == 1]

print("X: ", x)
print("")
print("xNew: ", xNew)