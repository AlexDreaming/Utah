import numpy as np
import matplotlib.pyplot as plt

f = open("result.txt", "r")
lines = f.readlines()
width = []
depth = []
mean = []
std = []

for line in lines:
    line = line.strip('[').strip('\n').strip(']').split(',')
    width.append(int(line[0]))
    depth.append(int(line[1]))
    mean.append(float(line[2]))
    std.append(float(line[3]))
depth_sort = sorted(depth)
f.close()
width = np.array(width).astype(int)
depth = np.array(depth).reshape(-1, 1)
mean = np.array(mean).reshape(-1, 1)
std = np.array(std).reshape(-1, 1)
print(width)
print('-'*100)
print(depth)
print('-'*100)
print(mean)
print('-'*100)
print(std)
index = np.arange(1, 50, 1, int).reshape(-1, 1)
pair = np.concatenate((depth, mean, std, index), axis=1)

# print(width)
# print('-'*100)
# print(depth)
# print('-'*100)
# print(mean)
# print('-'*100)
# print(std)
#print(pair[0][0])
# print('-'*100)
pair = pair[np.lexsort(pair[:, ::-1].T)]
print(pair)
#plt.errorbar(depth_sort, pair[:, 1], yerr=pair[:, 2], fmt='-o', ecolor='r')
plt.plot(depth_sort, np.log10(pair[:, 1]), color='#000000', marker='.', markerfacecolor='#929591', markersize=1, alpha=0.3)
plt.xlim((0, 42))
plt.ylim((-4.0, -1.8))
plt.xticks(np.arange(0, 45, 5))
plt.fill_between(depth_sort, np.log10(pair[:, 1]-pair[:, 2]), np.log10(pair[:, 1]+pair[:, 2]), facecolor='#000000',
                 alpha=0.3)
plt.xlabel('Depth', fontsize=18)
plt.ylabel('Log10(Error)', fontsize=18)
plt.text(1, -2, 'Width = 10', fontsize=15, ha='left')

cm = plt.cm.get_cmap('RdYlBu_r')
sc = plt.scatter(depth_sort, np.log10(pair[:, 1]), c=pair[:, 3], marker='s', s=100, cmap=cm, vmax=50)
cbar = plt.colorbar(sc)

cbar.set_label('Iter', size=15)

max = int(np.argmax(pair[:, 3]))
plt.scatter(depth_sort[max], np.log10(pair[:, 1][max]), color='', marker='s', edgecolors='#AD03DE', s=200)


plt.tight_layout()
plt.show()
