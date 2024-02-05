# Define FCM_effect function as needed for your implementation
# 得到一个与矩阵A形状相同的矩阵，具体功能还不确定，我觉得应该是随机得到一个矩阵A
def FCM_effect(A, n):
    return A * n.reshape(1, -1)