import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import save_figure_image
from fcm_nomal_3 import fcm
from FCM_effect_3 import FCM_effect
import rotate_x_labels
from scipy.special import expit  # Sigmoid function

# Load data
df = pd.read_csv('./data.csv')
A = df.values

A = A[:, 1:]  # Exclude headers
np.fill_diagonal(A, 0)
result = []
rep = 10000
Name = ['male_lamprey','female_lamprey','heron','salmon','sturgeon','trout','temperature','sunlight','oxygen'] # x轴标签
names = df.columns[1:]

num = len(names)


#这行代码计算了一个名为 nint 的变量。它使用 NumPy 的功能来统计矩阵 A 中非零元素的数量，并除以 A 的行数。这个计算的结果是矩阵 A 中每行的平均非零元素数量
nint = np.sum(A != 0) / A.shape[0]

# Specify indices
m_l_ind = names.get_loc('male_lamprey')
f_l_ind = names.get_loc('female_lamprey')
predator_ind = names.get_loc('predator')
prey_ind = names.get_loc('prey')
producer_ind = names.get_loc('producer')
oxygen_ind = names.get_loc('oxygen')
habitat_ind = names.get_loc('habitat')
sun_ind = names.get_loc('sunlight')
tempture_ind = names.get_loc('tempture')

inter = []
values = []
inter1 = []
values1 = []
inter2= []
values2 = []
inter3 = []
values3 = []
inter4 = []
values4 = []
inter5 = []
values5 = []
inter6 = []
values6 = []
inter7 = []
values7 = []
inter8 = []
values8 = []
inter9 = []
values9 = []

max = 0
i = 0
while i < rep:

    om = 1 - 0.1 ** np.random.uniform(1, 4)
    lambda_val = -1 / (nint * om) * np.log((1 - om) / om)
    i += 1
    Ar = lambda_val * np.random.rand(*A.shape) * A
    print(Ar)
    n, inte,value = fcm(Ar)
    inter.append(inte)
    if len(inter)>max:
        max = len(inter)
    values.append(value)
    ef = FCM_effect(Ar, n)
    cond1 = np.abs(ef[m_l_ind, prey_ind ]) > ef[f_l_ind, prey_ind ]
    cond2 = ef[m_l_ind, predator_ind] > ef[f_l_ind, predator_ind]
    if not(cond1 and cond2):
        continue

    n1, inte1 , value1 = fcm(Ar, 1 / 9)
    inter1.append(inte1)
    if len(inter1)>max:
        max = len(inter1)
    values1.append(value1)
    #print(values1)

    n2, inte2 , value2 = fcm(Ar, 2 / 8)
    inter2.append(inte2)
    if len(inter2)>max:
        max = len(inter2)
    values2.append(value2)

    n3, inte3 , value3 = fcm(Ar, 3 / 7)
    inter.append(inte3)
    if len(inter3)>max:
        max = len(inter3)
    values3.append(value3)

    n4, inte4 , value4 = fcm(Ar, 4 / 6)
    inter4.append(inte4)
    if len(inter4)>max:
        max = len(inter4)
    values4.append(value4)

    n5, inte5,value5 = fcm(Ar, 1)
    inter5.append(inte5)
    if len(inter5)>max:
        max = len(inter5)
    values5.append(value5)

    n6, inte6,value6 = fcm(Ar, 6 / 4)
    inter6.append(inte6)
    if len(inter6)>max:
        max = len(inter6)
    values6.append(value6)

    n7, inte7,value7 = fcm(Ar, 7 / 3)
    inter7.append(inte7)
    if len(inter7)>max:
        max = len(inter7)
    values7.append(value7)

    n8, inte8,value8 = fcm(Ar, 8 / 2)
    inter8.append(inte8)
    if len(inter8)>max:
        max = len(inter8)
    values8.append(value8)

    n9, inte9,value9 = fcm(Ar, 9/1)
    inter9.append(inte9)
    if len(inter9)>max:
        max = len(inter9)
    values9.append(value9)

max = 21
for matrix in values:
    while len(matrix)<max:
        matrix.append(matrix[-1])

for matrix in values1:
    while len(matrix)<max:
        matrix.append(matrix[-1])


for matrix in values2:
    while len(matrix)<max:
        matrix.append(matrix[-1])

for matrix in values3:
    while len(matrix)<max:
        matrix.append(matrix[-1])

for matrix in values4:
    while len(matrix)<max:
        matrix.append(matrix[-1])

for matrix in values5:
    while len(matrix)<max:
        matrix.append(matrix[-1])

for matrix in values6:
    while len(matrix)<max:
        matrix.append(matrix[-1])

for matrix in values7:
    while len(matrix)<max:
        matrix.append(matrix[-1])

for matrix in values8:
    while len(matrix)<max:
        matrix.append(matrix[-1])

for matrix in values9:
    while len(matrix)<max:
        matrix.append(matrix[-1])


vec01 = [0]*max
vec02 = [0]*max
for matrix in values:
    first_values_vector = np.concatenate([arr[0] for arr in matrix])
    second_values_vector = np.concatenate([arr[1] for arr in matrix])
    vec01 = (vec01 + first_values_vector)/2
    vec02 = (vec02 + second_values_vector)/2

vec11 = [0]*max
vec12 = [0]*max
for matrix in values1:
    first_values_vector = np.concatenate([arr[0] for arr in matrix])
    second_values_vector = np.concatenate([arr[1] for arr in matrix])
    vec11 = (vec11 + first_values_vector)/2
    vec12 = (vec12 + second_values_vector)/2

vec21 = [0]*max
vec22 = [0]*max
for matrix in values2:
    first_values_vector = np.concatenate([arr[0] for arr in matrix])
    second_values_vector = np.concatenate([arr[1] for arr in matrix])
    vec21 = (vec21 + first_values_vector)/2
    vec22 = (vec22 + second_values_vector)/2

vec31 = [0]*max
vec32 = [0]*max
for matrix in values3:
    first_values_vector = np.concatenate([arr[0] for arr in matrix])
    second_values_vector = np.concatenate([arr[1] for arr in matrix])
    vec31 = (vec31 + first_values_vector)/2
    vec32 = (vec32 + second_values_vector)/2

vec41 = [0]*max
vec42 = [0]*max
for matrix in values4:
    first_values_vector = np.concatenate([arr[0] for arr in matrix])
    second_values_vector = np.concatenate([arr[1] for arr in matrix])
    vec41 = (vec41 + first_values_vector)/2
    vec42 = (vec42 + second_values_vector)/2

vec51 = [0]*max
vec52 = [0]*max
for matrix in values5:
    first_values_vector = np.concatenate([arr[0] for arr in matrix])
    second_values_vector = np.concatenate([arr[1] for arr in matrix])
    vec51 = (vec51 + first_values_vector)/2
    vec52 = (vec52 + second_values_vector)/2

vec61 = [0]*max
vec62 = [0]*max
for matrix in values6:
    first_values_vector = np.concatenate([arr[0] for arr in matrix])
    second_values_vector = np.concatenate([arr[1] for arr in matrix])
    vec61 = (vec61 + first_values_vector)/2
    vec62 = (vec62 + second_values_vector)/2

vec71 = [0]*max
vec72 = [0]*max
for matrix in values7:
    first_values_vector = np.concatenate([arr[0] for arr in matrix])
    second_values_vector = np.concatenate([arr[1] for arr in matrix])
    vec71 = (vec71 + first_values_vector)/2
    vec72 = (vec72 + second_values_vector)/2

vec81 = [0]*max
vec82 = [0]*max
for matrix in values8:
    first_values_vector = np.concatenate([arr[0] for arr in matrix])
    second_values_vector = np.concatenate([arr[1] for arr in matrix])
    vec81 = (vec81 + first_values_vector)/2
    vec82 = (vec82 + second_values_vector)/2

vec91 = [0]*max
vec92 = [0]*max
for matrix in values9:
    first_values_vector = np.concatenate([arr[0] for arr in matrix])
    second_values_vector = np.concatenate([arr[1] for arr in matrix])
    vec91 = (vec91 + first_values_vector)/2
    vec92 = (vec92 + second_values_vector)/2

vec02 = vec02[:13]
vec12 = vec12[:13]
vec22 = vec22[:13]
vec32 = vec32[:13]
vec42 = vec42[:13]
vec52 = vec52[:13]
vec62 = vec62[:13]
vec72 = vec72[:13]
vec82 = vec82[:13]
vec92 = vec92[:13]
with open('资源.txt', 'w') as file:
    file.write("vec01: {}\n".format(vec02))
    file.write("vec11: {}\n".format(vec12))
    file.write("vec21: {}\n".format(vec22))
    file.write("vec31: {}\n".format(vec32))
    file.write("vec41: {}\n".format(vec42))
    file.write("vec51: {}\n".format(vec52))
    file.write("vec61: {}\n".format(vec62))
    file.write("vec71: {}\n".format(vec72))
    file.write("vec81: {}\n".format(vec82))
    file.write("vec91: {}\n".format(vec92))
plt.plot(range(1, 14), vec02, linestyle='-', color='#FF0000', label='nomal')
plt.plot(range(1, 14), vec12, linestyle='-', color='#0000FF', label='1:9')
plt.plot(range(1, 14), vec22, linestyle='-', color='#00FF00', label='2:8')
plt.plot(range(1, 14), vec32, linestyle='-', color='#FFA500', label='3:7')
plt.plot(range(1, 14), vec42, linestyle='-', color='#800080', label='4:6')
plt.plot(range(1, 14), vec52, linestyle='-', color='#00FFFF', label='5:5')
plt.plot(range(1, 14), vec62, linestyle='-', color='#FFFF00', label='6:4')
plt.plot(range(1, 14), vec72, linestyle='-', color='#FFC0CB', label='7:3')
plt.plot(range(1, 14), vec82, linestyle='-', color='#A52A2A', label='8:2')
plt.plot(range(1, 14), vec92, linestyle='-', color='#4B0082', label='9:1')
# 添加标题和标签
plt.title('Vector Plot')
plt.xlabel('Index')
plt.ylabel('Values')
plt.savefig('资源.png')
# 显示图例
plt.legend()

# 显示图形
plt.show()
