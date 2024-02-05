import numpy as np
from matplotlib import pyplot as plt

def fcm(A, n_fix=None):
    #定义激活函数
    def activation_function(x):
        x = np.asarray(x, dtype=float)
        return 1 / (1 + np.exp(-x))
    B = A.copy()
    n = np.random.rand(B.shape[0], 1)


    if n_fix is not None:
        n[1] = n_fix*n[0]

    flag = False
    counter = 0
    iterations_list = []
    n_values_list = []
    while not flag:
        # if n_fix is not None:
        #     n[1] = n_fix * n[0]
        counter += 1
        n_old = n.copy()
        n = activation_function(np.dot(B, n))
        if np.sum((n - n_old) ** 2) < 1e-6:
            flag = True
        if counter > 1000:
            flag = True
            n[:, 0] = np.nan

    B[-1, :] = -2
    flag = False
    if n_fix is not None:
        n[1] = n_fix*n[0]
    counter = 0
    while not flag:
        counter += 1
        n_old = n.copy()
        n = activation_function(np.dot(B, n))
        new_n = np.array([[np.sum(n[:4])],[np.sum(n[4:])]])
        iterations_list.append(counter)
        n_values_list.append(new_n.copy())
        if np.sum((n - n_old) ** 2) < 1e-6:
            flag = True

        if counter > 1000:
            flag = True
            n[:, 0] = np.nan
    return n,iterations_list,n_values_list

