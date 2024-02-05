import numpy as np

def fcm(A, n_fix=None):
    def activation_function(x):
       # x = np.array(x)
        x = np.asarray(x, dtype=float)  # 将x强制转换为float类型的NumPy数组
        #print(x)
        return 1 / (1 + np.exp(-x))

    # Initialize n vector (random)
    n = np.random.rand(A.shape[0], 1)
    # print(n_fix)
    # print(n)

    # Set fixed values if provided
    if n_fix is not None:
        n_fix=np.array(n_fix)
        n[n_fix[:, 0].astype(int)] = n_fix[:, 1][:, np.newaxis]*n[0]

    flag = False  # Set up a flag for the while loop
    counter = 0

    while not flag:
        counter += 1
        n_old = n.copy()  # Record current value of state
        n = activation_function(np.dot(A, n))  # Generate new value
        if n_fix is not None:
            n_fix = np.array(n_fix)
            n[n_fix[:, 0].astype(int)] = n_fix[:, 1][:, np.newaxis] * n[0]
        # Set fixed values if provided

        if np.sum((n - n_old) ** 2) < 1e-6:  # Check convergence
            flag = True  # If convergent, set flag true

        if counter > 1000:
            flag = True
            n[:, 0] = np.nan

    return n

