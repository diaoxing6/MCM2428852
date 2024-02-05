import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import save_figure_image
from fcm import fcm
from FCM_effect import FCM_effect
import rotate_x_labels
from scipy.special import expit  # Sigmoid function

# Load data
df = pd.read_csv('./data.csv')
A = df.values

A = A[:, 1:]  # Exclude headers
np.fill_diagonal(A, 0)


rep = 1000000
#Name = ['lamprey','heron','salmon','sturgeon','trout','temperature','sunlight','oxygen'] # x轴标签
names = df.columns[1:]

num = len(names)
n_all = np.zeros((num, rep))
n_1 = n_all.copy()  #0.56
n_2 = n_all.copy()  #0.78
n_3 = n_all.copy()
n_4 = n_all.copy()
n_5 = n_all.copy()
n_6 = n_all.copy()
n_7 = n_all.copy()
n_8 = n_all.copy()

#这行代码计算了一个名为 nint 的变量。它使用 NumPy 的功能来统计矩阵 A 中非零元素的数量，并除以 A 的行数。这个计算的结果是矩阵 A 中每行的平均非零元素数量
nint = np.sum(A != 0) / A.shape[0]

# Specify indices
lamprey_ind = names.get_loc('lamprey')
predator_ind = names.get_loc('predator')
prey_ind = names.get_loc('prey')
producer_ind = names.get_loc('producer')
oxygen_ind = names.get_loc('oxygen')
habitat_ind = names.get_loc('habitat')
sun_ind = names.get_loc('sunlight')
tempture_ind = names.get_loc('tempture')


i = 0
while i < rep:
    print(i)
    om = 1 - 0.1 ** np.random.uniform(1, 4)
    lambda_val = -1 / (nint * om) * np.log((1 - om) / om)
    i += 1

    Ar = lambda_val * np.random.rand(*A.shape) * A

    n = fcm(Ar)
    ef = FCM_effect(Ar, n)
    #print([[cat_ind, 0]])
    n1 = fcm(Ar, [[lamprey_ind,0]])   #
    n2 = fcm(Ar, [[predator_ind, 0]])   #
    n3 = fcm(Ar,[[prey_ind,0]])
    n4 = fcm(Ar,[[producer_ind,0]])
    n5 = fcm(Ar,[[oxygen_ind,0]])
    n6 = fcm(Ar,[[habitat_ind,0]])
    n7 = fcm(Ar,[[sun_ind,0]])
    n8 = fcm(Ar,[[tempture_ind,0]])

   # cond3 = ef[cat_ind, bb_ind] > ef[rat_ind, bb_ind]

    if np.all(np.isfinite([n, n1, n2,n3,n4,n5,n6,n7,n8])):
        n_all[:, i - 1] = n[:, 0]
        n_1[:, i - 1] = n1[:, 0]
        n_2[:, i - 1] = n2[:, 0]
        n_3[:, i - 1] = n3[:, 0]
        n_4[:, i - 1] = n4[:, 0]
        n_5[:, i - 1] = n5[:, 0]
        n_6[:, i - 1] = n6[:, 0]
        n_7[:, i - 1] = n7[:, 0]
        n_8[:, i - 1] = n8[:, 0]
    else:
        i -= 1

data = [n_all[:, 0], n_1[:, 0], n_2[:, 0], n_3[:, 0], n_4[:, 0], n_5[:, 0], n_6[:, 0], n_7[:, 0], n_8[:, 0]]

plt.figure(figsize=(10, 6))  # 设置图形的大小
plt.boxplot(data, patch_artist=True, notch=True,
            labels=['1', '2', '3','4','5','6','7','8','9'])  # 绘制箱形图

# 添加图形的标题和轴标签
plt.title('Species Richness under Different Conditions')
plt.ylabel('Species Richness')
plt.xlabel('Condition')

plt.savefig('result100w')
# 显示图形
plt.show()

# c_outcomes = np.mean(n_c / n_all > 1, axis=1)  #只删了1个
# b_outcomes = np.mean(n_b / n_all > 1, axis=1)  #两者都删了
# print(c_outcomes)
# print(b_outcomes)

# c_ratio = np.median(n_c / n_all - 1, axis=1)
# c_05 = np.percentile(n_c / n_all - 1, 5, axis=1)
# c_95 = np.percentile(n_c / n_all - 1, 95, axis=1)
# b_ratio = np.median(n_b / n_all - 1, axis=1)
# b_05 = np.percentile(n_b / n_all - 1, 5, axis=1)
# b_95 = np.percentile(n_b / n_all - 1, 95, axis=1)
# c_abs = np.mean(n_c - n_all, axis=1)
# b_abs = np.mean(n_b - n_all, axis=1)
#
# max_diff = np.percentile(n_b / n_all - 1, 100, axis=1) / np.percentile(n_b / n_all - 1, 0, axis=1)
#
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
# x = np.arange(1, 10)
# fig, ax = plt.subplots(figsize=(10, 6))
# width = 0.35  # 柱状图的宽度
# rects1 = ax.bar(x - width/2, c_ratio*100, width, label='sex-ratio 0.56:0.44', color='blue')
# rects2 = ax.bar(x + width/2, b_ratio*100, width, label='sex-ratio 0.78:0.22', color='orange')
# ax.set_ylabel('Percentage change\nin abundance')
# ax.set_xticks(x)
# ax.set_xticklabels(Name, rotation=-15) # 旋转x轴标签
# ax.legend()
# plt.savefig('增减频率1.png')
# # 显示图形
# plt.show()
#
# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
# x = np.arange(1, 10)
# fig, ax = plt.subplots(figsize=(10, 6))
# width = 0.35  # 柱状图的宽度
# rects1 = ax.bar(x - width/2, c_outcomes*100, width, label='sex-ratio 0.56:0.44', color='blue')
# rects2 = ax.bar(x + width/2, b_outcomes*100, width, label='sex-ratio 0.78:0.22', color='orange')
# ax.set_ylabel('Frequency of increase')
# ax.set_xticks(x)
# ax.set_xticklabels(Name, rotation=-15) # 旋转x轴标签
# ax.legend()
# plt.savefig('百分比1.png')
# # 显示图形
# plt.show()


#
# name = ['male_lamprey','female_lamprey','heron','salmon','sturgeon','trout','temperature','sunlight','oxygen'] # x轴标签
# num = len(c_outcomes) # 数据组数量
# col1 = 'skyblue' # 示例颜色
# col2 = 'orange' # 示例颜色
# lw = 1.05 # 线宽
#
# # 创建图形和轴对象
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # 绘制堆叠柱形图
# bars = plt.bar(np.arange(num), c_ratio*100, color=col1, label='sex-ratio 0.56:0.44', linewidth=lw, edgecolor=col1)
# bars2 = plt.bar(np.arange(num), b_ratio*100, bottom=c_ratio*100, color=col2, label='sex-ratio 0.78:0.22', linewidth=lw, edgecolor=col2)
#
# # 设置图例和y轴标签
# plt.legend(loc='best')
# plt.ylabel('Percentage change\nin abundance')
#
# # 设置x轴标签
# ax.set_xticks(np.arange(num))
# ax.set_xticklabels(name, rotation=-15) # 旋转x轴标签
#
# # 设置x轴范围
# plt.xlim(0.5 - 1, num + 0.5 - 1)
#
# # 调整图形大小（此函数在Matplotlib中不存在，因此省略）
#
# # 保存图形
# plt.savefig('增减频率2.png', dpi=300) # DPI可根据需要调整
#
# plt.show() # 显示图形
#
# # # 显示图形
# # plt.show()
#
# names = ['male_lamprey','female_lamprey','heron','salmon','sturgeon','trout','temperature','sunlight','oxygen'] # x轴标签
# num = len(c_outcomes) # 数据组数量
# col1 = 'skyblue' # 示例颜色
# col2 = 'orange' # 示例颜色
# lw = 1.05 # 线宽
#
# # 创建图形和轴对象
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # 绘制堆叠柱形图
# bars = plt.bar(np.arange(num), c_outcomes*100, color=col1, label='sex-ratio 0.56:0.44' , linewidth=lw, edgecolor=col1)
# bars2 = plt.bar(np.arange(num), b_outcomes*100, bottom=c_outcomes*100, color=col2, label='sex-ratio 0.78:0.22', linewidth=lw, edgecolor=col2)
#
# # 设置图例和y轴标签
# plt.legend(loc='best')
# plt.ylabel('% of increase')
#
# # 设置x轴标签
# ax.set_xticks(np.arange(num))
# ax.set_xticklabels(names, rotation=-15) # 旋转x轴标签
#
# # 设置x轴范围
# plt.xlim(0.5 - 1, num + 0.5 - 1)
#
# # 调整图形大小（此函数在Matplotlib中不存在，因此省略）
#
# # 保存图形
# plt.savefig('百分比2.png', dpi=300) # DPI可根据需要调整
#
# plt.show() # 显示图形