import numpy as np
import pandas as pd
from itertools import product
pd.option_context('format.precision', 3)

"""
The code to transform and sort the data used for fig 9 and 10.
"""
n_starts = 10
nr_starts = np.arange(0,n_starts,1)
n_points=20
c_fixed = ['t0']
folder = f"/data/numerik/people/abankowski/neuro_param_estimation/codes/analysis/experiments_differential_evolution/Est_2/fixed_params_{c_fixed[0]}_fig9multistart_pointsadded"
animals = [1,2,3,4,5]

if (c_fixed == ['L']):
	fixed_param_grid = np.log10(np.linspace(1,2.5,n_points))
	lst = list(product(fixed_param_grid,nr_starts))
if (c_fixed == ['gP']):
	fixed_param_grid = np.log10(np.linspace(1,200,n_points))
	lst = list(product(fixed_param_grid,nr_starts))
if (c_fixed == ['k_base']):
	fixed_param_grid  = np.log10(np.concatenate(([0.5,1,2,3,4],np.linspace(5,75,n_points-5))))
	lst = list(product(fixed_param_grid,nr_starts))
if (c_fixed == ['kR']):
	fixed_param_grid = np.log10(np.linspace(1,50,n_points))
	lst = list(product(fixed_param_grid,nr_starts))
if (c_fixed == ['nves']):
	#fixed_param_grid=np.log10([10,20])
	fixed_param_grid = np.log10(np.linspace(1,50,20,dtype=int))
	lst = list(product(fixed_param_grid,nr_starts))
if (c_fixed == ['t0']):
	fixed_param_grid = np.log10(np.concatenate((np.linspace(10**(-2),10**(-1.5),4)[1:-1],np.linspace(10**(-1.5),1e-1,10)[1:-1])))
	#fixed_param_grid = np.log10(np.logspace(-9,-1,17))
	lst = list(product(fixed_param_grid,nr_starts))						 #np.log10(np.logspace(-2,-1,3)[1:])#np.log10(np.logspace(-9,-2,15))
if (c_fixed == ['N_var']):
	fixed_param_grid  = np.log10(np.linspace(20,500,20,dtype=int))##
	lst = list(product(fixed_param_grid,nr_starts))	
if (c_fixed == ['gV']):
	fixed_param_grid = np.log10(np.linspace(0.25,1,4)[:3])
	#fixed_param_grid = np.log10(np.linspace(1,25,n_points))
	lst = list(product(fixed_param_grid,nr_starts))	
if (c_fixed == ['nsites']):
	fixed_param_grid = np.log10(np.linspace(1,200,n_points,dtype=int))
	lst = list(product(fixed_param_grid,nr_starts))	
	
fixed_param_grid=np.around(fixed_param_grid,decimals=3)

#load the data

np.set_printoptions(precision=3)
all_data = {f'animal {i}':np.around(np.load(f"{folder}/a{i}/result.npy"),decimals=3) for i in animals}



#sort them by the fixed parameter value and find the minimum loss

for i in animals:
	optimal_values=[]
	df = pd.DataFrame(all_data[f'animal {i}'])
	df.columns=[f'{c_fixed[0]}','1','2','3','4','5','6','7','8','loss']
	for j in range(len(fixed_param_grid)):
		dfopt = df.loc[df[f'{c_fixed[0]}']==fixed_param_grid[j]]
		minind = dfopt.index[np.argmin(dfopt['loss'].values, axis=0)]
		optimal_values= np.append([optimal_values],np.array([df.iloc[minind,:]]))

	optimal_values=optimal_values.reshape(len(fixed_param_grid),10)
	np.save(f"{folder}/a{i}/optimal_values.npy",optimal_values)
