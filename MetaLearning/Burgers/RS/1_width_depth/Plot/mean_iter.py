import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

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
f.close()
width = np.array(width)
depth = np.array(depth)
mean = np.array(mean)
std = np.array(std)
print(width)
print('-'*100)
print(depth)
print('-'*100)
print(mean)
print('-'*100)
print(std)

fig, ax = plt.subplots()


cm = plt.cm.get_cmap('RdYlBu_r')

index = np.arange(1, 51, 1, int)
sc = ax.scatter(depth, width, s=75, c=index, alpha=0.8, cmap=cm)
ax.set_xticks(np.arange(2, 9, 1))
ax.set_yticks(np.arange(20, 65, 5))
cbar = plt.colorbar(sc)
cbar_range = np.arange(0, 51, 10)
cbar.set_ticks(cbar_range)
labels = np.round(cbar_range)
cbar.set_ticklabels(labels)
cbar.set_label('Iter', size=15)

# plt.xlim(1, 9)
plt.ylim(18,62)
ax.set_xlabel(r'Depth', fontsize=15)
ax.set_ylabel(r'Width', fontsize=15)
ax.set_title('Distribution',size=15)
ax.grid(True)

max = np.argmin(mean)
ax.scatter(depth[max], width[max], color='', marker='s', edgecolors='#AD03DE', s=200)

fig.tight_layout()
plt.show()
