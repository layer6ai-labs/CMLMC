import sys
import os

modelname = sys.argv[1]
end = int(sys.argv[2])

modelfolder = 'results/checkpoints/'+modelname+'/'

for j in range(1, end+1):
    cpname = modelfolder + 'checkpoint{}.pt'.format(j)
    command = 'rm {}'.format(cpname)
    try:
        os.system(command)
    except:
        pass