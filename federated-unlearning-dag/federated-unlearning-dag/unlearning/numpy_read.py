import numpy as np
import json


test = np.load('datasets/purchase/purchase2_train.npy', allow_pickle=True)
test = test.reshape((1,))[0]
print(test)