from matplotlib import pyplot as plt
import os
import sys
sys.path.insert(0, '/data/numerik/people/abankowski/neuro_param_estimation/codes')
import numpy as np
from opt_settings import *
from opt_parfor_parallel import *
plt.rcParams['axes.linewidth']=0.4
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
   "font.serif": "cm",
    "axes.prop_cycle": cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', 
            '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']),
    })

run_dir_plotting = '/data/numerik/people/abankowski/neuro_param_estimation/codes/analysis/experiments_differential_evolution/Est_2/Est_2_test'

#collect all best params
animals = range(1,6)
params = np.load("/data/numerik/people/abankowski/neuro_param_estimation/codes/analysis/experiments_differential_evolution/Est_2/Est_2_test/bestparams.npy")


avg_p = np.load("/data/numerik/people/abankowski/neuro_param_estimation/codes/analysis/experiments_differential_evolution/Est_2/Est_2_test/avgparams.npy")
#np.save(f'{run_dir_plotting}/all_best_params.npy',params)
#np.save(f'{run_dir_plotting}/avg_params.npy',avg_p)
print('Averaged parameters:','\n',param_combo,'\n',10**avg_p)

#plot of the baseline functions and average
fig,ax = plt.subplots(figsize=(5,2.5),dpi=200,constrained_layout=True)
for i in range(0,5):
    L=10**params[i,4]
    k_base = 10**params[i,7]
    t0 = 10**params[i,6]
    baseline = 10**(L*(1-exp(-k_base*((t*variables['fac']-variables['t_wait'])-t0))))
    ax.plot(times[0:2500],baseline[0:2500],label=f'animal {i+1}', linewidth=1)
#nominal values in baseline fct
L_nom=0.95
k_base_nom = 12
t0_nom = 0.096
baseline_nom=  10**(L_nom*(1-exp(-k_base_nom*(t-t0_nom))))
ax.plot(times[0:2500], baseline_nom[0:2500], label='nominal', lw=1,linestyle='dashed', c='green')

#plot the averaged baseline fct
L_avg = 10**avg_p[4]
k_base_avg = 10**avg_p[7]
t0_avg = 10**avg_p[6]
avg_baseline = 10**(L_avg*(1-exp(-k_base_avg*((t*variables['fac']-variables['t_wait'])-t0_avg))))
ax.plot(times[0:2500], avg_baseline[0:2500], label='averaged', lw=1.5,linestyle='dotted', c='black')

#labellings
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_yticks(np.array([0,20,40]))
#ax.title('Model fit')
ax.set_xlabel('$t$ in $s$', fontsize=15)
ax.set_ylabel('$f_{\mathrm{baseline}}$ in $s^{-1}$', fontsize=15)
ax.legend(bbox_to_anchor=(1, 1.05),ncols=1, fontsize=12)
fig.savefig(f'{run_dir_plotting}/baseline.png')