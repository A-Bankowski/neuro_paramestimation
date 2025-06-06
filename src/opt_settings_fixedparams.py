from numpy import log10
import os
import time
import numpy as np
from copy import deepcopy
from itertools import product

n_starts = 10
nr_starts = np.arange(0,n_starts,1)
pool_size=120
opt_options={'disp':True, 'maxiter':500}
seed=None #100
initial_radius_factor = 0.85 #0.5 # parameters are initialized randomly in the interval [(1-initial_radius_factor)*nom,(1+initial_radius_factor)*nom]
NF_weight=1e-28 #1e-28 #1e-25 #1e-28

param_est = 2

animal = 1

objective = 'peaks_cur' 

measurements = 'real'

param_combo_optimized = 'c'

fixed_param_combo = 'c_fixed'

#c_fixed = ['nsites']
#c_fixed = ['nves']
#c_fixed = ['N_var']
#c_fixed = ['gV']
#c_fixed = ['gP']
#c_fixed = ['kR']
#c_fixed = ['L']
#c_fixed = ['k_base']
c_fixed = ['t0']


"""
Path directories for saving
"""
run_name = f"{param_est}_{n_starts}_starts"

yr, month, day, hr, minute = map(int, time.strftime("%Y %m %d %H %M").split())

start_time = f'{yr}{month}{day}_{hr}{minute}'

def my_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
folder = 'analysis'
folder2='experiments_differential_evolution' #'new_results_l2_dist_more_info'
folder3 = f'fixed_params_{c_fixed[0]}_fig9multistart_pointsadded'
#my_mkdir(folder)
#my_mkdir(folder2)
my_mkdir(f"{folder}/{folder2}/Est_{param_est}/{folder3}")
run_dir=f"{folder}/{folder2}/Est_{param_est}/{folder3}/a{animal}"


my_mkdir(run_dir)
################################################################################################################







#number of points to vary the fixed parameter on and determine these points via a grid
n_points = 20
#diff gv bzw steps

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
	lst = list(product(fixed_param_grid,nr_starts))	
if (c_fixed == ['nsites']):
	fixed_param_grid = np.log10(np.linspace(1,200,n_points,dtype=int))
	lst = list(product(fixed_param_grid,nr_starts))	

number_of_sites=1

param_list=list(('N_var', 't_wait', 'gV', 'gP', 'fac', 'sigma', 'L', 'k_base', 't0', 'a1', 'mu1', 'a2', 'mu2', 'a3', 'mu3', 'a4', 'mu4', 'a5', 'mu5', 'a6', 'mu6', 'a7', 'mu7', 'a8', 'mu8', 'a9', 'mu9', 'a10', 'mu10', 'a11', 'mu11', 'a12', 'mu12', 'a13', 'mu13', 'a14', 'mu14', 'a15', 'mu15', 'a16', 'mu16', 'a17', 'mu17', 'a18', 'mu18', 'a19', 'mu19', 'a20', 'mu20', 'a21', 'mu21', 'a22', 'mu22', 'a23', 'mu23', 'a24', 'mu24', 'a25', 'mu25', 'a26', 'mu26', 'a27', 'mu27', 'a28', 'mu28', 'a29', 'mu29', 'a30', 'mu30', 'a31', 'mu31', 'a32', 'mu32', 'a33', 'mu33', 'a34', 'mu34', 'a35', 'mu35', 'a36', 'mu36', 'a37', 'mu37', 'a38', 'mu38', 'a39', 'mu39', 'a40', 'mu40', 'a41', 'mu41', 'a42', 'mu42', 'a43', 'mu43', 'a44', 'mu44', 'a45', 'mu45', 'a46', 'mu46', 'a47', 'mu47', 'a48', 'mu48', 'a49', 'mu49', 'a50', 'mu50', 'a51', 'mu51', 'a52', 'mu52', 'a53', 'mu53', 'a54', 'mu54', 'a55', 'mu55', 'a56', 'mu56', 'a57', 'mu57', 'a58', 'mu58', 'a59', 'mu59', 'a60', 'mu60', 'kUmax', 'kUmin', 'steep', 'cliffstart', 'kR', 'nves', 'nsites'))
param_list_indices={param_list[i]:i for i in range(len(param_list))}

print(param_list_indices)

param_bounds_dict={'N_var':(1,5000), 
                    'gV':(1e-8,5000),
                    'gP':(1e-8,5000),
                   #'L':(1e-8,np.log10(5000)),
                    'L':(1e-8,500),
                    'kR':(1e-8,5000),
                    'nves': (1,100),
                    'nsites': (1,100),
                    't0':(1e-8,1.0),
                    'fac':(1e-8,10.0),
                    'k_base':(1e-8,500)}


c42 = ['N_var','gV','gP','kR','L','nves','k_base','t0','nsites']


c = [param for param in c42 if param not in c_fixed]

c.remove('nsites')
param_combo_dict={"c42":c42,"c_fixed":c_fixed,'c':c}

param_combo = param_combo_dict[param_combo_optimized]
param_indices = [param_list_indices[p] for p in param_combo]
param_bounds_list=[[np.log10(param_bounds_dict[par][0]), np.log10(param_bounds_dict[par][1])] for par in param_combo]
param_bounds = tuple([tuple(pb) for pb in param_bounds_list])

fixed_param_combo = param_combo_dict[fixed_param_combo]
fixed_param_index = [param_list_indices[p] for p in fixed_param_combo]

param_combo = param_combo_dict[param_combo_optimized]
print(" Estimation ", param_est, " Animal ", animal, " fixed parameter ", c_fixed,"\n")
print("param_combo:", param_combo,"param_indices:",param_indices, "\n param_bounds (log):", param_bounds)
check= [param_list[p] for p in param_indices]
print("check",check)