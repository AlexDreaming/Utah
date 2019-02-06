import numpy as np
import matplotlib.pyplot as plt

f = open("Poisson_Ntr.txt", "r")
lines = f.readlines()

Ntr = []
mean = []
std = []

for line in lines:
    line = line.strip('[').strip('\n').strip(']').split(',')
    Ntr.append(int(line[0]))
    mean.append(float(line[1]))
    std.append(float(line[2]))

f.close()
Ntr = np.array(Ntr)
mean = np.array(mean)
std = np.array(std)

print(Ntr)
print('-'*100)
print(mean)
print('-'*100)
print(std)


#plt.errorbar(depth_sort, pair[:, 1], yerr=pair[:, 2], fmt='-o', ecolor='r')
plt.plot(Ntr, mean, color='b', marker='s', markerfacecolor='b', markersize=8)
# plt.xlim((1000, 110000))
plt.ylim((0.00005, 0.00065))
#plt.xticks([50, 100, 200, 300, 400, 456])
plt.fill_between(Ntr, mean-std, mean+std, facecolor='#8F99FB')
plt.xlabel('Ntr of Poisson', fontsize=18)
plt.ylabel('Error', fontsize=18)
#plt.text(350, 0.032, 'Nf=10000', fontsize=18, ha='left')
plt.title("Sampling Error (Width=41 Depth=3)", fontsize=18)
plt.tight_layout()
plt.show()
