#%matplotlib inline
import time

import numpy 				 as np


A = np.zeros(5)
B = np.zeros(5)
C = np.ones(5)

logical = np.array([True, True, False])

print((A == B).any() and (A == C).any())
print(np.sum(logical))