import numpy as np

folder ='/data/numerik/people/abankowski/neuro_param_estimation/codes/analysis/experiments_differential_evolution/Est_1/Est_1_test'
animals = [1,2,3,4,5]
a = np.array([ np.load(f"{folder}/a{i}/result.npy")
               for i in animals])

amin = [ np.argmin(a[i-1,:,-1]) for i in animals]

bestparams = np.array([a[i-1,amin[i-1],1:-1] for i in animals])
meanvalues = np.mean(bestparams,axis=0)
optimal_loss = np.array([a[i-1,amin[i-1],-1] for i in animals])
np.save(f"{folder}/optimal_loss.npy", optimal_loss)
np.save(f"{folder}/bestparams.npy", bestparams)
np.save(f"{folder}/avgparams.npy", meanvalues)