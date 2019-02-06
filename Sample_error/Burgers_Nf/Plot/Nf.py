import numpy as np
import matplotlib.pyplot as plt

f = open("Burgers_Nf.txt", "r")
lines = f.readlines()

Nf = []
mean = []
std = []

for line in lines:
    line = line.strip('[').strip('\n').strip(']').split(',')
    Nf.append(int(line[0]))
    mean.append(float(line[1]))
    std.append(float(line[2]))

f.close()
Nf = np.array(Nf)
mean = np.array(mean)
std = np.array(std)

print(Nf)
print('-'*100)
print(mean)
print('-'*100)
print(std)

num = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#plt.errorbar(depth_sort, pair[:, 1], yerr=pair[:, 2], fmt='-o', ecolor='r')
plt.plot(num, mean, color='b', marker='s', markerfacecolor='b', markersize=8)
#plt.xlim((100, 110000))
#plt.ylim((-0.0005, 0.0065))
plt.xticks(num, ['0.5', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.fill_between(num, mean-std, mean+std, facecolor='#8F99FB')
plt.xlabel(r'Nf($\times$e4) of Burgers', fontsize=18)
plt.ylabel('Error', fontsize=18)
plt.text(8, 0.0026, 'Nu=456', fontsize=18, ha='left')
plt.title("Sampling Error (Width=46 Depth=8)", fontsize=18)
plt.tight_layout()
plt.show()
