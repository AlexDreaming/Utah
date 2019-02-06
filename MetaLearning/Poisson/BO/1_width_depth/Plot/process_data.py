for i in range(100):
    t = 993232+i
    f = open("slurm-"+str(t)+".out")
    line = f.readline()
    i = 1
    while line:
        line = f.readline()
        i += 1
        if i == 12:
            t = open("result.txt", "a")
            t.write(line)
            t.close()
    f.close()
