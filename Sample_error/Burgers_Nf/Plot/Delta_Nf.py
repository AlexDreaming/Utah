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

# print(Nf)
# print('-'*100)
print(mean)
print('-'*100)
# print(std)

t = list([0.535136441478128])  #Nf = 10
for i in mean:
    t.append(i)
delta_err = (mean - t[:-1])[1:]
delta_err = np.array(delta_err)
print(delta_err)

num = np.arange(1, 11, 1)


plt.plot(num, delta_err, color='#000000', marker='.', markerfacecolor='#929591', markersize=1, alpha=0.3)
plt.xlim((0, 11))
# plt.plot(np.linspace(0, 10, 100), [0]*100, 'g')
# plt.ylim((-0.0005, 0.0065))
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x_label = ['1e4', '2e4', '3e4', '4e4', '5e4', '6e4', '7e4', '8e4', '9e4', '10e4']
plt.xticks(x)
# plt.fill_between(Nf, mean-std, mean+std, facecolor='#8F99FB')
plt.xlabel(r'Nf($\times$e4) of Burgers', fontsize=18)
plt.ylabel('$\Delta$Error', fontsize=18)
#plt.text(85000, 0.0023, 'Nu=456', fontsize=18, ha='left')
plt.title("Sampling Error (Width=46 Depth=8)", fontsize=18)

cm = plt.cm.get_cmap('RdYlBu_r')
sc = plt.scatter(num, delta_err, c=mean[1:], marker='s', s=100, cmap=cm)
cbar = plt.colorbar(sc)
cbar.set_label('Error', size=15)

plt.tight_layout()
plt.show()
