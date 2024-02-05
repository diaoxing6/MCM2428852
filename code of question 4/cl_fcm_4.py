import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import save_figure_image
from fcm_4 import fcm
from FCM_effect_4 import FCM_effect
import rotate_x_labels
from scipy.special import expit  # Sigmoid function

# Load data
df = pd.read_csv('./data.csv')
A = df.values

A = A[:, 1:]  # Exclude headers
np.fill_diagonal(A, 0)


rep = 100
Name = ['male_lamprey','female_lamprey','parasite','producer','trout','sturgeon','salmon','temperature','sunlight','oxygen'] # x轴标签
names = df.columns[1:]

num = len(names)
n_all = np.zeros((num, rep))
n_c = n_all.copy()  #0.56
n_b = n_all.copy()  #0.78


#这行代码计算了一个名为 nint 的变量。它使用 NumPy 的功能来统计矩阵 A 中非零元素的数量，并除以 A 的行数。这个计算的结果是矩阵 A 中每行的平均非零元素数量
nint = np.sum(A != 0) / A.shape[0]

# Specify indices
m_l_ind = names.get_loc('male_lamprey')
f_l_ind = names.get_loc('female_lamprey')
parasite_ind = names.get_loc('parasite')
producer_ind = names.get_loc('producer')
salmon_ind = names.get_loc('salmon')
sturgeon_ind = names.get_loc('sturgeon')
trout_ind = names.get_loc('trout')
temperature_ind = names.get_loc('temperature')
sun_ind = names.get_loc('sunlight')
oxygen_ind = names.get_loc('oxygen')


i = 0
while i < rep:
    om = 1 - 0.1 ** np.random.uniform(1, 4)
    lambda_val = -1 / (nint * om) * np.log((1 - om) / om)
    i += 1

    Ar = lambda_val * np.random.rand(*A.shape) * A
    print(Ar)

    n = fcm(Ar)

    ef = FCM_effect(Ar, n)
    #print([[cat_ind, 0]])
    nc = fcm(ef, [[f_l_ind,0.44/0.56]])   #
    nb = fcm(ef, [[f_l_ind, 0.22/0.78]])   #

    cond1 = np.abs(ef[m_l_ind, trout_ind ]) > ef[f_l_ind, trout_ind ]
    cond2 = ef[m_l_ind, parasite_ind] > ef[f_l_ind, parasite_ind]
    cond3 = ef[m_l_ind, trout_ind] > ef[m_l_ind,salmon_ind]
    cond4 = ef[f_l_ind, trout_ind] > ef[f_l_ind,salmon_ind]
    cond5 = ef[trout_ind,m_l_ind ] < ef[trout_ind,f_l_ind]
    cond6 = ef[salmon_ind,m_l_ind] < ef[salmon_ind,f_l_ind]
   # cond3 = ef[cat_ind, bb_ind] > ef[rat_ind, bb_ind]

    if np.all(np.isfinite([n, nc, nb])) and cond1 and cond2 and cond3 and cond4 and cond5 and cond6:
        n_all[:, i - 1] = n[:, 0]
        n_c[:, i - 1] = nc[:, 0]
        n_b[:, i - 1] = nb[:, 0]
    else:
        i -= 1

c_outcomes = np.mean(n_c / n_all > 1, axis=1)  #只删了1个
b_outcomes = np.mean(n_b / n_all > 1, axis=1)  #两者都删了

c_ratio = np.median(n_c / n_all - 1, axis=1)
c_05 = np.percentile(n_c / n_all - 1, 5, axis=1)
c_95 = np.percentile(n_c / n_all - 1, 95, axis=1)
b_ratio = np.median(n_b / n_all - 1, axis=1)
b_05 = np.percentile(n_b / n_all - 1, 5, axis=1)
b_95 = np.percentile(n_b / n_all - 1, 95, axis=1)
c_abs = np.mean(n_c - n_all, axis=1)
b_abs = np.mean(n_b - n_all, axis=1)

#max_diff = np.percentile(n_b / n_all - 1, 100, axis=1) / np.percentile(n_b / n_all - 1, 0, axis=1)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
x = np.arange(1, 11)
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.35  # 柱状图的宽度

rects1 = ax.bar(x - width/2, c_ratio*100, width, label='sex-ratio 0.56:0.44', color='blue')
rects2 = ax.bar(x + width/2, b_ratio*100, width, label='sex-ratio 0.78:0.22', color='orange')
ax.set_ylabel('Percentage change\nin abundance')
ax.set_xticks(x)
ax.set_xticklabels(Name, rotation=-15) # 旋转x轴标签
ax.legend()
plt.savefig('增减频率1.png')
# 显示图形
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
x = np.arange(1, 11)
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.35  # 柱状图的宽度

rects1 = ax.bar(x - width/2, c_outcomes*100, width, label='sex-ratio 0.56:0.44', color='#007ACC')
rects2 = ax.bar(x + width/2, b_outcomes*100, width, label='sex-ratio 0.78:0.22', color='#FFA500')
ax.set_ylabel('Frequency of increase')
ax.set_xticks(x)
ax.set_xticklabels(Name, rotation=-15) # 旋转x轴标签
ax.legend()
plt.savefig('百分比1.png')
# 显示图形
plt.show()



name = ['male_lamprey','female_lamprey','parasite','producer','trout','sturgeon','salmon','temperature','sunlight','oxygen'] # x轴标签
num = len(c_outcomes) # 数据组数量
lw = 1.05 # 线宽

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制堆叠柱形图
bars = plt.bar(np.arange(num), c_ratio*100, color='#007ACC', label='sex-ratio 0.56:0.44', linewidth=lw, edgecolor='#007ACC')
bars2 = plt.bar(np.arange(num), b_ratio*100, bottom=c_ratio*100, color='#FFA500', label='sex-ratio 0.78:0.22', linewidth=lw, edgecolor='#FFA500')

# 设置图例和y轴标签
plt.legend(loc='best')
plt.ylabel('Percentage change\nin abundance')

# 设置x轴标签
ax.set_xticks(np.arange(num))
ax.set_xticklabels(name, rotation=-15) # 旋转x轴标签

# 设置x轴范围
plt.xlim(0.5 - 1, num + 0.5 - 1)

# 调整图形大小（此函数在Matplotlib中不存在，因此省略）

# 保存图形
plt.savefig('增减频率2.png', dpi=300) # DPI可根据需要调整

plt.show() # 显示图形

# # 显示图形
# plt.show()

names = ['male_lamprey','female_lamprey','parasite','producer','trout','sturgeon','salmon','temperature','sunlight','oxygen'] # x轴标签
num = len(c_outcomes) # 数据组数量
col1 = 'skyblue' # 示例颜色
col2 = 'orange' # 示例颜色
lw = 1.05 # 线宽

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制堆叠柱形图
bars = plt.bar(np.arange(num), c_outcomes*100, color='#007ACC', label='sex-ratio 0.56:0.44' , linewidth=lw, edgecolor=col1)
bars2 = plt.bar(np.arange(num), b_outcomes*100, bottom=c_outcomes*100, color='#FFA500', label='sex-ratio 0.78:0.22', linewidth=lw, edgecolor=col2)

# 设置图例和y轴标签
plt.legend(loc='best')
plt.ylabel('% of increase')

# 设置x轴标签
ax.set_xticks(np.arange(num))
ax.set_xticklabels(names, rotation=-15) # 旋转x轴标签

# 设置x轴范围
plt.xlim(0.5 - 1, num + 0.5 - 1)

# 调整图形大小（此函数在Matplotlib中不存在，因此省略）

# 保存图形
plt.savefig('百分比2.png', dpi=300) # DPI可根据需要调整

plt.show() # 显示图形