from matplotlib import pyplot as plt
import os
import sys
#sys.path.insert(0, '/data/numerik/people/abankowski/neuro_param_estimation/codes')
import numpy as np
from opt_settings import *
from opt_parfor_parallel import *
from matplotlib.pyplot import cm

plt.rcParams['axes.linewidth']=0.4
#folder = "/data/numerik/people/abankowski/neuro_param_estimation/codes/analysis/experiments_differential_evolution/Est_3/pop_size30"
animal = 1
popsize=30
run_name = "1"
number_starts = 1
figsize = (5,3)
#run_dir_plotting=f"{folder}/a{animal}"
run_dir_plotting =f'/data/numerik/people/abankowski/neuro_param_estimation/codes/analysis/experiments_differential_evolution/Est_{run_name}/Est_{run_name}_test/a{animal}'
#some plotting settings

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
   "font.serif": "cm",
    "axes.prop_cycle": cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', 
            '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']),
    })

def labelling(key):
    if key == 'gV':
        return'$g_V$'
    elif key == 'gP':
        return '$g_P$'
    elif key== 'kR':
        return '$k_R$'
    elif key == 'nves':
        return '$n_{\mathit{ves}}$'
    elif key == 'N_var':
        return '$N$'
    elif key=='k_base':
        return '$m_1$'
    elif key=='L':
        return '$L$'
    elif key == 't0':
        return '$m_2$'
    else:
        print('No corresponding parameter found')
color = cm.tab10(np.linspace(0, 1, 10))#[[0,1,2,3,4]]
#plots currents all
animals = [1,2,3,4,5]
current_data_strings = ['data/Current_data_animal'+str(i)+'.npy' for i in animals]
fig,ax = plt.subplots(figsize=(5,6),nrows=5,sharex=True, sharey=True, constrained_layout=True)
for i in animals:
    current_data_plot = np.load(current_data_strings[i-1])
    ax[i-1].plot(times,10**9 *current_data_plot, label=f'{i}',c=color[i-1])
fig.supxlabel('$t$ in $s$',fontsize=20)
#ax[4].set_ylabel('Current $C^{\mathrm{data}} in $n$A', fontsize=16)
    #ax.legend(loc='lower right',ncols=2, fontsize=13.7)
plt.yticks([-100,0])
fig.legend(title='animal',title_fontsize=15,loc='upper right', ncols=5, fontsize=11,bbox_to_anchor=(1.,1.1))
fig.supylabel('Current $C^{\mathrm{data}}$ in $n$A',fontsize=20)
fig.savefig("/data/numerik/people/abankowski/neuro_param_estimation/codes/analysis/data_plot.png", bbox_inches='tight')

#load the optimized parameters and the loss of the run
optimized_params = np.load(f'{run_dir_plotting}/result.npy')
#loss = f_obj(optimized_params)
obj_values = optimized_params[:,-1]


#sorting
sorted_args = np.argsort(obj_values)
sorted_obj = np.sort(obj_values)




###########################
######## Fit plots ########
###########################


#min_ind = obj_values.index(min(obj_values))
param_Id = list(model.getParameterIds()[i] for i in param_indices) #necessary? 
model.setParameterById(dict(zip(param_Id, optimized_params[sorted_args[0],1:-1])))
all_params = model.getParameters() #updated parameters(?)
#zoom options for the plots
zoom_name = ['full','zoomed_a','zoomed_b','zoomed_c','zoomed_d','zoomed_e','zoomed_f']
#plot the fits for the best results in the multi-start optimization
for fit in ['Current']:
    j=0
    for (n1,n) in [(0,3000),(0,500),(500,1000),(1000,1500),(1500,2000),(2000,2500),(2500,3000)]:
        #data = NF_obs
        param_Id = list(model.getParameterIds()[i] for i in param_indices)
        model.setParameterById(dict(zip(param_Id, optimized_params[sorted_args[0],1:-1])))
        all_params = model.getParameters()
        plt.figure(dpi=350)
        if fit=='Current':
            plt.plot(times[n1:n],current(NF(np.array(all_params)))[n1:n],'-',markersize = 2.5,linewidth = 1.5,label='model')
            plt.plot(times[n1:n],current_data[n1:n],linewidth=0.9,label='data',c='grey')
        elif fit=='NF':
            plt.plot(times[n1:n],NF(np.array(all_params))[n1:n],'-',markersize = 2.5,linewidth = 1.5,label='model')
            plt.plot(times[n1:n],NF_data[n1:n],linewidth=0.9,label='data')           
        param_Id = list(model.getParameterIds()[i] for i in param_indices)
        #plt.title('Model fit')
        plt.xlabel('Time in $s$')
        plt.ylabel('Current in $(n\mathrm{A})²$')
        plt.legend()
        plt.savefig(f'{run_dir_plotting}/best_fit_plot_{fit}_{zoom_name[j]}.png')
        j+=1


#best fit plot with zoom in for param est 2 and 3
if param_est==1:
    param_Id = list(model.getParameterIds()[i] for i in param_indices)
    model.setParameterById(dict(zip(param_Id, optimized_params[sorted_args[0],1:-1])))
    all_params = model.getParameters()
    fig, ax = plt.subplots(figsize=figsize,dpi=200,constrained_layout=True)
    if animal == 1:
        ax.set_ylim(-120,5)
        ax.set_xlim(-0.001,1)
        ax.plot(times[0:2500],current(NF(np.array(all_params)))[0:2500]*10**9,'-',
                markersize = 0.3,linewidth = 1.,label='$C^{\mathrm{sim}}_{\hat{\\theta}}$', c='grey')
        ax.plot(times[0:2500],current_data[0:2500]*10**9,linewidth=1.,label='$C^{\mathrm{data}}$', c='tab:blue')
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        ax.set_yticks(np.array([-100,-75,-50,-25,0]))

        ax.set_xlabel('$t$ in $s$', fontsize=16)
        ax.set_ylabel('Current in $n$A', fontsize=16)
        ax.legend(loc='lower right',ncols=2, fontsize=14)
        fig.savefig(f'{run_dir_plotting}/best_fit_plot_current_est{run_name}_a{animal}.png')
    elif animal ==4:
        ax.set_ylim(-120,5)
        ax.set_xlim(-0.001,1)
        ax.plot(times[0:2500],current(NF(np.array(all_params)))[0:2500]*10**9,'-',
                markersize = 3,linewidth = 1.,label='$C^{\mathrm{sim}}_{\hat{\\theta}}$', c='grey')
        ax.plot(times[0:2500],current_data[0:2500]*10**9,linewidth=1.,label='$C^{\mathrm{data}}$', c='tab:red')
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        ax.set_yticks(np.array([-100,-75,-50,-25,0]))
        #ax.title('Model fit')
        ax.set_xlabel('$t$ in $s$', fontsize=16)
        ax.set_ylabel('Current in $n$A', fontsize=16)
        ax.legend(loc='lower right',ncols=2, fontsize=14)
        fig.savefig(f'{run_dir_plotting}/best_fit_plot_current_est{run_name}_a{animal}.png')
elif param_est==2 or param_est==3:
    param_Id = list(model.getParameterIds()[i] for i in param_indices)
    model.setParameterById(dict(zip(param_Id, optimized_params[sorted_args[0],1:-1])))
    all_params = model.getParameters()
    fig, ax = plt.subplots(figsize=figsize,dpi=200,constrained_layout=True)
    if animal == 1:
        ax.set_ylim(-120,30)
        ax.set_xlim(-0.001,1)
        ax.plot(times[0:2500],current(NF(np.array(all_params)))[0:2500]*10**9,'-',
                markersize = 0.3,linewidth = 1,label='$C^{\mathrm{sim}}_{\hat{\\theta}}$', c='grey')
        ax.plot(times[0:2500],current_data[0:2500]*10**9,
                linewidth=1,label='$C^{\mathrm{data}}$', c='tab:blue')
        axin = ax.inset_axes([0.3,0.68,0.5,0.3])
        axin.set_xlim(0.2,0.3)
        axin.set_ylim(-100,-25)
        axin.plot(times[0:2500],current(NF(np.array(all_params)))[0:2500]*10**9, linewidth=1,
                  c='tab:blue', label='$C^{\mathrm{sim}}_{\hat{\\theta}}$')
        axin.plot(times[0:2500],current_data[0:2500]*10**9,  linewidth=1,c='grey', label='$C^{\mathrm{data}}$')
        ax.indicate_inset_zoom(axin, edgecolor="black", lw=1)
        axin.set_yticklabels([])
        axin.set_xticklabels([])
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        ax.set_yticks(np.array([-100,-75,-50,-25,0]))
        #ax.title('Model fit')
        ax.set_xlabel('$t$ in $s$', fontsize=16)
        ax.set_ylabel('Current in $n$A', fontsize=16)
        ax.legend(loc='lower right',ncols=2, fontsize=13.7)
        fig.savefig(f'{run_dir_plotting}/best_fit_plot_current_est{run_name}_a{animal}.png')
    elif animal ==4:
        ax.set_ylim(-120,30)
        ax.set_xlim(-0.001,1)
        ax.plot(times[0:2500],current(NF(np.array(all_params)))[0:2500]*10**9,'-',
                markersize = 3,linewidth = 1,label='$C^{\mathrm{sim}}_{\hat{\\theta}}$', c='grey')
        ax.plot(times[0:2500],current_data[0:2500]*10**9,linewidth=1,label='$C^{\mathrm{data}}$', c='tab:red')
        axin = ax.inset_axes([0.4,0.75,0.5,0.23])
        axin.set_xlim(0.2,0.3)
        axin.set_ylim(-65,-10)
        axin.plot(times[0:2500],current(NF(np.array(all_params)))[0:2500]*10**9, linewidth=1,c='tab:red', label='$C^{\mathrm{sim}}_{\hat{\\theta}}$')
        axin.plot(times[0:2500],current_data[0:2500]*10**9,  linewidth=1,c='grey', label='$C^{\mathrm{data}}$')
        ax.indicate_inset_zoom(axin, edgecolor="black", lw=1)
        axin.set_yticklabels([])
        axin.set_xticklabels([])
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        ax.set_yticks(np.array([-100,-75,-50,-25,0]))
        #ax.title('Model fit')
        ax.set_xlabel('$t$ in $s$', fontsize=16)
        ax.set_ylabel('Current in $n$A', fontsize=16)
        ax.legend(loc='lower right',ncols=2, fontsize=13.7)
        fig.savefig(f'{run_dir_plotting}/best_fit_plot_current_w_zoom_est{run_name}_a{animal}.png')
#
#variability plots (fig A2 etc)
if param_est ==1 or param_est==3:
    fig, ax = plt.subplots(figsize=(4,2.5), dpi=200, constrained_layout=True)
    ax.plot(np.array(param_bounds),
            ['N','$g_V$','$g_P$','$k_R$','$n_{\mathrm{ves}}$'],
              linestyle='dashed', c='black',lw=0.8)
    [ax.plot(optimized_params[sorted_args[i],1:-1].T,
             ['N','$g_V$','$g_P$','$k_R$','$n_{\mathrm{ves}}$'], 
             linestyle='solid', marker='o', c='grey',lw=1,markersize=0.5,alpha=1-i/40) 
             for i in range(6,optimized_params.shape[0])]
    [ax.plot(optimized_params[sorted_args[i],1:-1].T,
             ['N','$g_V$','$g_P$','$k_R$','$n_{\mathrm{ves}}$'], 
             lw=1,c='tab:orange',linestyle='solid', marker='o',markersize=0.5,alpha=1-i/40) 
             for i in range(1,6)]
    ax.plot(optimized_params[sorted_args[0],1:-1].T,
            ['N','$g_V$','$g_P$','$k_R$','$n_{\mathrm{ves}}$'],
              lw=1,c='red',linestyle='solid', marker='o',markersize=0.5)
    ax.set_yticks(['N','$g_V$','$g_P$','$k_R$','$n_{\mathrm{ves}}$'])
    ax.set_xticks([0,2])
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_xlabel('$log$10 of parameter values', fontsize=15)
    ax.set_ylabel('parameters', fontsize=15)
    fig.savefig(f'{run_dir_plotting}/variability_of_parameters{run_name}_a{animal}.png')
elif param_est ==2:
    fig, ax = plt.subplots(figsize=(4,2.5), dpi=200, constrained_layout=True)

    #swap columns for correct display
    ##swap L and nves column (4 and 5) and t0 and kbase (6 and 7)
    dummy1 = optimized_params[:,1:-1].T[[5,4]]
    dummy2 = optimized_params[:,1:-1].T[[7,6]]
    population_parameters = np.concatenate((optimized_params[:,1:-1].T[0:4,:], dummy1,dummy2))
    dum1 = np.array(param_bounds)[[5,4]]
    dum2 = np.array(param_bounds)[[7,6]]
    param_bounds = np.concatenate((np.array(param_bounds)[0:4,:],dum1,dum2))
    dummy1 = optimized_params[sorted_args[0],1:-1][[5,4]]
    dummy2 = optimized_params[sorted_args[0],1:-1][[7,6]]
    optimized_params = np.concatenate((optimized_params[sorted_args[0],1:-1].T[0:4], dummy1,dummy2))


    ax.plot(np.array(param_bounds),
            ['N','$g_V$','$g_P$','$k_R$','$n_{\mathrm{ves}}$','L','$m_1$','$m_2$'],
              linestyle='dashed', c='black',lw=0.8)
    
    [ax.plot(population_parameters.T[sorted_args[i],:],
             ['N','$g_V$','$g_P$','$k_R$','$n_{\mathrm{ves}}$','L','$m_1$','$m_2$'], 
             lw=1,c='tab:orange',linestyle='solid', marker='o',markersize=0.5,alpha=1-i/40) for i in range(1,6)]
    [ax.plot(population_parameters.T[sorted_args[i],:],
             ['N','$g_V$','$g_P$','$k_R$','$n_{\mathrm{ves}}$','L','$m_1$','$m_2$'], 
             linestyle='solid', marker='o', c='grey',lw=1 ,markersize=0.5,alpha=1-i/40)
               for i in range(6,population_parameters.shape[1])]
    ax.plot(optimized_params,
        ['N','$g_V$','$g_P$','$k_R$','$n_{\mathrm{ves}}$','L','$m_1$','$m_2$'], 
        lw=1,c='red',linestyle='solid', marker='o',markersize=0.3)
    ax.set_yticks(['N','$g_V$','$g_P$','$k_R$','$n_{\mathrm{ves}}$','L','$m_1$','$m_2$'])
    ax.set_xticks([-8,0,2])
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_xlabel('log10 of parameter values', fontsize=15)
    ax.set_ylabel('parameters', fontsize=15)
    fig.savefig(f'{run_dir_plotting}/variability_of_parameters{run_name}_a{animal}.png')

#alpha=i/population_parameters.shape[1]
#, alpha=i/population_parameters.shape[1]