from numpy import log10
import os
import time
import numpy as np
from copy import deepcopy

n_starts = 100
pool_size=120
opt_options={'disp':True, 'maxiter':500}
seed=None #100
initial_radius_factor = 0.85 #0.5 # parameters are initialized randomly in the interval [(1-initial_radius_factor)*nom,(1+initial_radius_factor)*nom]
NF_weight=1e-28 #1e-28 #1e-25 #1e-28

param_est = 1

animal = 4

objective = 'peaks_cur' 

measurements = 'real'

#animal=5



#exp directory to load initial parameters and best index
if param_est ==1:
    param_combo_optimized = 'c2'
    if animal ==1:
         exp_dir = '/data/numerik/people/abankowski/neuro_param_estimation/data/optimization/grp_2a/a1_c2_1_sites_real_40_starts_peaks_NF_cur_Nelder-Mead_1702899591_1e-28/initial_params_array.npy'
         ind = 26   
    elif animal == 2:
        exp_dir = '/data/numerik/people/abankowski/neuro_param_estimation/data/optimization/grp_2a/a2_c2_1_sites_real_40_starts_peaks_NF_cur_Nelder-Mead_1702899605_1e-28/initial_params_array.npy'
        ind = 36
    elif animal == 3:
        exp_dir = '/data/numerik/people/abankowski/neuro_param_estimation/data/optimization/grp_2a/a3_c2_1_sites_real_40_starts_peaks_NF_cur_Nelder-Mead_1702901407_1e-28/initial_params_array.npy'
        ind = 30 #2
    elif animal == 4:
        exp_dir = '/data/numerik/people/abankowski/neuro_param_estimation/data/optimization/grp_2a/a4_c2_1_sites_real_40_starts_peaks_NF_cur_Nelder-Mead_1702899707_1e-28/initial_params_array.npy'
        ind = 26
    elif animal == 5:
        exp_dir = '/data/numerik/people/abankowski/neuro_param_estimation/data/optimization/grp_2a/a5_c2_1_sites_real_40_starts_peaks_NF_cur_Nelder-Mead_1702899720_1e-28/initial_params_array.npy'
        ind = 16
elif param_est ==2:
    param_combo_optimized = 'c42'
    if animal == 1:
        exp_dir = '/data/numerik/people/abankowski/neuro_param_estimation/data/optimization/grp_1/a1/initial_params_array.npy'
        ind = 37
    elif animal ==2:
        exp_dir = '/data/numerik/people/abankowski/neuro_param_estimation/data/optimization/grp_1/a2_c42_1_sites_real_40_starts_peaks_NF_cur_Nelder-Mead_1702918247_1e-28/initial_params_array.npy'
        ind = 11
    elif animal ==3:
        exp_dir = '/data/numerik/people/abankowski/neuro_param_estimation/data/optimization/grp_1/a3_c42_1_sites_real_40_starts_peaks_NF_cur_Nelder-Mead_1702918232_1e-28/initial_params_array.npy'
        ind = 3
    elif animal ==4:
        exp_dir = '/data/numerik/people/abankowski/neuro_param_estimation/data/optimization/grp_1/a4_c42_1_sites_real_40_starts_peaks_NF_cur_Nelder-Mead_1702918216_1e-28/initial_params_array.npy'
        ind = 25
    elif animal ==5:
        exp_dir = '/data/numerik/people/abankowski/neuro_param_estimation/data/optimization/grp_1/a5_c42_1_sites_real_40_starts_peaks_NF_cur_Nelder-Mead_1702918197_1e-28/initial_params_array.npy'
        ind = 29

elif param_est == 3:
    param_combo_optimized = 'c2'
    if animal ==1:
        exp_dir = '/data/numerik/people/abankowski/neuro_param_estimation/data/optimization/grp_2g/a1_c2_1_sites_real_40_starts_peaks_NF_cur_Nelder-Mead_1704812879_1e-28/initial_params_array.npy'
        ind = 13
    elif animal ==2:
        exp_dir = '/data/numerik/people/abankowski/neuro_param_estimation/data/optimization/grp_2g/a2_c2_1_sites_real_40_starts_peaks_NF_cur_Nelder-Mead_1704812864_1e-28/initial_params_array.npy'
        ind = 4
    elif animal ==3:
        exp_dir = '/data/numerik/people/abankowski/neuro_param_estimation/data/optimization/grp_2g/a3_c2_1_sites_real_40_starts_peaks_NF_cur_Nelder-Mead_1704812846_1e-28/initial_params_array.npy'
        ind = 10
    elif animal ==4:
        exp_dir = '/data/numerik/people/abankowski/neuro_param_estimation/data/optimization/grp_2g/a4_c2_1_sites_real_40_starts_peaks_NF_cur_Nelder-Mead_1704812827_1e-28/initial_params_array.npy'
        ind = 8
    elif animal ==5:
        exp_dir = '/data/numerik/people/abankowski/neuro_param_estimation/data/optimization/grp_2g/a5_c2_1_sites_real_40_starts_peaks_NF_cur_Nelder-Mead_1704812734_1e-28/initial_params_array.npy'
        ind = 25




number_of_sites =  1
L_ = 1.6 #1.7 #
t0_ = 3.9e-4 #7.1e-5 #
k_base_ = 18.7 #40.6 #
N_var_ = 334
kR_ = 15.2
gP_ = 94

run_name = f"{param_combo_optimized}_{n_starts}_starts"

opt_algotrithm = 'Nelder-Mead'

#opt_algotrithm = 'Powell'

yr, month, day, hr, minute = map(int, time.strftime("%Y %m %d %H %M").split())

start_time = f'{yr}{month}{day}_{hr}{minute}'

def my_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
folder = 'analysis' #'new_results_l2_dist_more_info'

my_mkdir(folder)


run_dir=f"{folder}/a{animal}_{run_name}_{start_time}"


my_mkdir(run_dir)



param_list=list(('N_var', 't_wait', 'gV', 'gP', 'fac', 'sigma', 'L', 'k_base', 't0', 'a1', 'mu1', 'a2', 'mu2', 'a3', 'mu3', 'a4', 'mu4', 'a5', 'mu5', 'a6', 'mu6', 'a7', 'mu7', 'a8', 'mu8', 'a9', 'mu9', 'a10', 'mu10', 'a11', 'mu11', 'a12', 'mu12', 'a13', 'mu13', 'a14', 'mu14', 'a15', 'mu15', 'a16', 'mu16', 'a17', 'mu17', 'a18', 'mu18', 'a19', 'mu19', 'a20', 'mu20', 'a21', 'mu21', 'a22', 'mu22', 'a23', 'mu23', 'a24', 'mu24', 'a25', 'mu25', 'a26', 'mu26', 'a27', 'mu27', 'a28', 'mu28', 'a29', 'mu29', 'a30', 'mu30', 'a31', 'mu31', 'a32', 'mu32', 'a33', 'mu33', 'a34', 'mu34', 'a35', 'mu35', 'a36', 'mu36', 'a37', 'mu37', 'a38', 'mu38', 'a39', 'mu39', 'a40', 'mu40', 'a41', 'mu41', 'a42', 'mu42', 'a43', 'mu43', 'a44', 'mu44', 'a45', 'mu45', 'a46', 'mu46', 'a47', 'mu47', 'a48', 'mu48', 'a49', 'mu49', 'a50', 'mu50', 'a51', 'mu51', 'a52', 'mu52', 'a53', 'mu53', 'a54', 'mu54', 'a55', 'mu55', 'a56', 'mu56', 'a57', 'mu57', 'a58', 'mu58', 'a59', 'mu59', 'a60', 'mu60', 'kUmax', 'kUmin', 'steep', 'cliffstart', 'kR', 'nves', 'nsites'))
param_list_indices={param_list[i]:i for i in range(len(param_list))}
#print(param_lisinitt)
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

c_p = ['gV','nves']

c1=['N_var','gV','gP','kR']
c2=c1+['nves']
c3=c2+['nsites']
c4=c1+['L']
c5=c4+['nves']
c6=c5+['nsites']

c7=c2+['nsites']

c8=c5+['t0','k_base','t0','fac']
c9=c6+['t0','k_base','t0','fac']


c11=c1+ ['t0']
c12 = c11 + ['k_base']
c41 = c5 + ['t0']
c42 = c41+ ['k_base']
c43 = c42[:]
c43.remove('N_var')
c44 = c5+['k_base']
c45 = c42[:]
c45.remove('L')
c46 = c45[:]
c46.remove('k_base')
c47 = c45[:]
c47.remove('t0')



param_combo_dict={"c1":c1,"c2":c2,"c3":c3,"c4":c4,"c5":c5,"c6":c6,"c7":c7,"c8":c8,"c9":c9, "c11":c11,"c12":c12, "c41":c41, "c42":c42, "c43":c43,"c44":c44,"c45":c45,"c46":c46,"c47":c47,"c_p":c_p}

param_combo = param_combo_dict[param_combo_optimized]
param_indices = [param_list_indices[p] for p in param_combo]
param_bounds_list=[[np.log10(param_bounds_dict[par][0]), np.log10(param_bounds_dict[par][1])] for par in param_combo]
param_bounds = tuple([tuple(pb) for pb in param_bounds_list])


param_combo = param_combo_dict[param_combo_optimized]

print("param_combo:", param_combo,"param_indices:",param_indices, "\n param_bounds (log):", param_bounds)
check= [param_list[p] for p in param_indices]
print("check",check)