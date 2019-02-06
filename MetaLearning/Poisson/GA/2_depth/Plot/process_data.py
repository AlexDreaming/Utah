for i in range(50):
    t = 912340+i
    f = open("slurm-"+str(t)+".out")
    line = f.readline()
    i = 1
    while line:
        line = f.readline()
        i += 1
        if i == 13:
            print(line)
    f.close()
