from matplotlib import pyplot as plt
import os
import sys
#sys.path.insert(0, '/data/numerik/people/abankowski/neuro_param_estimation/codes')
import numpy as np
from opt_settings import *
from opt_parfor_parallel import *

animal = 1
popsize=30
run_name = "3"
number_starts = 1
both = False
run_dir_plotting = f"/data/numerik/people/abankowski/neuro_param_estimation/codes/results_paper/Param_Est_{run_name}_test_correct"
run_dir_plotting_DE =f'/data/numerik/people/abankowski/neuro_param_estimation/codes/analysis/experiments_differential_evolution/Est_{run_name}/Est_{run_name}_test_correct'
run_dir_plotting_NM =f'/data/numerik/people/abankowski/neuro_param_estimation/codes/results_paper/Param_Est_{run_name}/Nelder_Mead'

#some plotting settings

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
   "font.serif": "cm",
    "axes.prop_cycle": cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', 
            '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']),
    })

#load the optimized parameters and the loss of the DE rzb
optimized_params_DE = np.load(f'{run_dir_plotting_DE}/a{animal}/result.npy')
obj_values_DE = optimized_params_DE[:,-1]#np.load(f'{run_dir_plotting_DE}/a{animal}/population_loss.npy')

#population_parameters_DE = np.load(f"{run_dir_plotting_DE}/a{animal}/population.npy")

#sorting
sorted_args_DE = np.argsort(obj_values_DE)
sorted_obj_DE = np.sort(obj_values_DE)

# load the paramters of Nelder Mead
#optimized_params_NM = np.load(f'{run_dir_plotting_NM}/Nelder_Mead/a{animal}_{run_name}_1000/best_params.npy')
#population_parameters_NM = np.load(f'{run_dir_plotting_NM}/Nelder_Mead/a{animal}_{run_name}_1000/optimized_params_array.npy')#
obj_values_NM = np.load(f"{run_dir_plotting_NM}/a{animal}_{run_name}_1000/all_losses.npy")#np.array([f_obj(population_parameters_NM[i]) for i in range(population_parameters_NM.shape[0])])#
#np.save(f'{run_dir_plotting}/Nelder_Mead/a{animal}_{run_name}_1000/all_losses.npy',obj_values_NM)
sorted_args_NM = np.argsort(obj_values_NM)
sorted_obj_NM = np.sort(obj_values_NM)

############################
###### Waterfallplot #######
############################
if both == True:
    fig, ax = plt.subplots(dpi=350,figsize=(5,3),constrained_layout=True)

            

    ax.plot(range(1,len(sorted_obj_NM)+1),sorted_obj_NM,'.',markersize=5, label='Nelder Mead')
    xticks=[1]+list(range(0,len(sorted_obj_NM)+1,(len(sorted_obj_NM)+1)//5)[1:])
    ax.set_ylim(min([sorted_obj_DE[0],sorted_obj_NM[0]])-1000,sorted_obj_NM[-1])
    ax.set_xticks(xticks)
    #ax.set_yticks([625,15000,30000])  
    ax.plot(range(1,len(sorted_obj_DE)+1),sorted_obj_DE,'.',markersize=5, c='tab:orange', label='Differential Evolution')
    axin = ax.inset_axes([0.2,0.67,0.5,0.3])
    axin.set_xlim(-2,int(len(sorted_args_DE)+1))
    axin.set_ylim(min([sorted_obj_DE[0],sorted_obj_NM[0]])-100,sorted_obj_DE[int(len(sorted_obj_DE)-1)]+50)
    axin.plot(range(1,int(len(sorted_obj_DE))+1),sorted_obj_DE,'.',markersize=5, c='tab:orange', label='Differential Evolution')
    axin.plot(range(1,len(sorted_obj_DE)+1),sorted_obj_NM[:len(sorted_obj_DE)],'.',markersize=5, c='tab:blue', label='Nelder Mead')

    ax.indicate_inset_zoom(axin, edgecolor="black", lw=0.5)
    axin.yaxis.set_tick_params(labelsize=10)
    #axin.set_yticks(np.linspace(int(sorted_obj_DE[0]-1),int(sorted_obj_DE[int(len(sorted_obj_DE)-1)]+1),3))
    axin.set_xticks([])
    #ax.set_yticks(np.linspace(int(sorted_obj_DE[0]-10),int(sorted_obj_NM[-1]+10),3, dtype=int))#
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.set_xlabel('Sorted run index', fontsize=15)
    ax.set_ylabel("$\\chi²(\hat{\\theta})$ in ($n\mathrm{A})^{2}$" , fontsize=15)
    ax.legend(loc='upper right', ncols=2, bbox_to_anchor=(1,1.2), fontsize=11)
    fig.savefig(f'{run_dir_plotting_DE}/waterfall_plot_{run_name}_{animal}_both.png')

else:
    fig, ax = plt.subplots(dpi=350,figsize=(5,3),constrained_layout=True)
    xticks=[1]+list(range(0,len(sorted_obj_DE)+1,(len(sorted_obj_DE)+1)//5)[1:])
    ax.set_xticks(xticks)
    #ax.set_yticks([625,15000,30000])  
    ax.plot(range(1,len(sorted_obj_DE)+1),sorted_obj_DE,'.',markersize=5, c='tab:blue', label='Differential Evolution')
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.set_xlabel('Sorted run index', fontsize=15)
    ax.set_ylabel("$\\chi²(\hat{\\theta})$ in ($n\mathrm{A})^{2}$" , fontsize=15)
    #ax.legend(loc='upper right', ncols=2, bbox_to_anchor=(1,1.2), fontsize=11)
    fig.savefig(f'{run_dir_plotting_DE}/waterfall_plot_{run_name}_{animal}.png')