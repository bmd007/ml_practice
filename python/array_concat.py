import numpy as np

degrees = np.asarray([1, 3, 6, 9])
predict = np.zeros((1, 3))
res = np.zeros((4, 4))

for i in range(0, len(degrees)):
    # res[i, : ] = np.concatenate((degrees[i], predict), axis=None)
    res[i, : ] =  np.asarray([1, 3, 6, 9])

print(res)

