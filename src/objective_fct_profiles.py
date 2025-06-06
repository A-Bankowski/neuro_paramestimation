import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.pyplot import cm
from cycler import cycler
from itertools import product

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
   "font.serif": "cm",
    "axes.prop_cycle": cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', 
            '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']),
    })


"""
This code generates fig 9 and 10. For the correct shape of the data, run fig9datamnaged.py first.
"""
fixed_param=['N_var','gV','gP','kR','nves','L','k_base','t0']
animals =[1,2,3,4,5]
lst = list(product(fixed_param,animals))
#lst.remove(('nves', 1))
colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink']

folder = f"/data/numerik/people/abankowski/neuro_param_estimation/codes/analysis/experiments_differential_evolution/Est_2"
optim_params = np.load(f"/data/numerik/people/abankowski/neuro_param_estimation/codes/analysis/experiments_differential_evolution/Est_2/Est_2_test/bestparams.npy")
optimal_loss = np.load(f"/data/numerik/people/abankowski/neuro_param_estimation/codes/analysis/experiments_differential_evolution/Est_2/Est_2_test/optimal_loss.npy")
run_dirs = {key:f"{folder}/fixed_params_{key[0]}_fig9multistart/a{key[1]}" for i,key in enumerate(lst)}


all_combo= ['N_var','gV','gP','kR','L','nves','t0','k_base','loss']
optimum = [np.concatenate((optim_params[i-1][:],[optimal_loss[i-1]])) for i in animals]
df_opt = pd.DataFrame(optimum)
df_opt.columns=all_combo
dataframe_collection ={}
letters=['$\mathrm{(a)}$','$\mathrm{(b)}$','$\mathrm{(c)}$','$\mathrm{(d)}$',
         '$\mathrm{(e)}$'
         ,'$\mathrm{(f)}$','$\mathrm{(g)}$','$\mathrm{(h)}$']
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

        

#the plot for all
fig, ax = plt.subplots(figsize=(15,8), nrows=2,ncols=4, constrained_layout = True, sharey=True)

for i,key in enumerate(lst):
    index_list= [0,1,2,3,4,5,6,7]
    fixed_params, animal = key
    
    #load the data corresponding to the fixed parameter and the animal
    
    data = np.load(f'{run_dirs[(fixed_params,animal)]}/optimal_values.npy')[:,[0,2,3,4,5,6,7,8,9]]
    #as the fixed parameter was added at the beginning we need to rearrange the column heads
    all_combos= ['N_var','gV','gP','kR','L','nves','t0','k_base','loss']
    all_combos_optim = ['N_var','gV','gP','kR','L','nves','t0','k_base','loss']
    fixed_ind = all_combos_optim.index(fixed_params)
    index_list.remove(fixed_ind)
    temp_arr = np.concatenate((np.array([optim_params[animal-1][fixed_ind]]),
                                        optim_params[animal-1][index_list],
                                        np.array([optimal_loss[animal-1]])))


    data = np.concatenate((data,np.array([temp_arr])))
    #as the fixed parameter was added at the beginning we need to rearrange the column heads
    
    column_head = all_combos

    column_head.remove(fixed_params)
    column_head = [fixed_params]+column_head
    #load into a dataframe
    df = pd.DataFrame(i for i in data)
    #add the column heads
    df.columns= column_head
    df=df.sort_values(by=[fixed_params])
    #collect all dataframes in a dict just in case
    dataframe_collection[f"({fixed_params},{animal})"]=df
    plot_index = fixed_param.index(fixed_params)

    #the actual plotting -- 
    ax[plot_index//4,plot_index%4].plot(10**df[fixed_params],df['loss']/optimal_loss[animal-1],
                                         'go-', markersize=2.5,c=colours[animal-1],
                                           label=f'animal {animal}')
    ax[plot_index//4,plot_index%4].hlines(3.71,10**df[fixed_params].iloc[0],
                                          10**df[fixed_params][-1:],ls='dashed',
                                          colors='grey',label='1+$\Delta_{1,0.9}$',
                                          lw=2.5)
    ax[plot_index//4,plot_index%4].set_ylim(0,4.5)
    ax[plot_index//4,plot_index%4].set_yticks([0,1,2,3])
    ax[plot_index//4,plot_index%4].set_xlabel(labelling(fixed_params), fontsize=30)
    if (plot_index//4,plot_index%4)==(0,0) or (plot_index//4,plot_index%4)==(1,0):
        ax[plot_index//4,plot_index%4].set_ylabel('Norm. profile', fontsize=25)
    ax[plot_index//4,plot_index%4].xaxis.set_tick_params(labelsize=25)
    ax[plot_index//4,plot_index%4].yaxis.set_tick_params(labelsize=25)
    if fixed_params=='t0':
        #ax[plot_index//4,plot_index%4].set_xscale('log')
        ax[plot_index//4,plot_index%4].set_xticks([0,0.1])
for i in range(len(ax.flatten())):
    ax.flatten()[i].text(-0.15,1.08,letters[i],transform = ax.flatten()[i].transAxes, fontsize=25)

line,labels=ax[0,0].get_legend_handles_labels()
fig.legend(line[:10][::2]+[line[1]],labels[:10][::2]+[labels[1]],loc='upper right', ncols=6, fontsize=20,bbox_to_anchor=(1,1.1))
fig.savefig(f"{folder}/fig9objective.png", bbox_inches='tight')
"""
#plot only one parameter fixed
fig, ax = plt.subplots(figsize=(4,2.6), constrained_layout = True)


for i in animals:
    fixed_params, animal = (fixed_param[0],i)
    index_list= [0,1,2,3,4,5,6,7]
    all_combos = ['N_var','gV','gP','kR','L','nves','t0','k_base','loss']
    all_combos_optim = ['N_var','gV','gP','kR','L','nves','t0','k_base','loss']
    #load the data corresponding to the fixed parameter and the animal
    data = np.load(f'{run_dirs[(fixed_params,i)]}/optimal_values.npy')[:,[0,2,3,4,5,6,7,8,9]]#np.load('/data/numerik/people/abankowski/neuro_param_estimation/codes/analysis/experiments_differential_evolution/Est_2/fixed_params_N_var_test6/a1/result.npy')
    #add optimal parameter array
    fixed_ind = all_combos_optim.index(fixed_params)
    index_list.remove(fixed_ind)
    temp_arr = np.concatenate((np.array([optim_params[animal-1][fixed_ind]]),
                                        optim_params[animal-1][index_list],
                                        np.array([optimal_loss[animal-1]])))


    data = np.concatenate((data,np.array([temp_arr])))
    #as the fixed parameter was added at the beginning we need to rearrange the column heads
    
    column_head = all_combos

    column_head.remove(fixed_params)
    column_head = [fixed_params]+column_head
    #load into a dataframe
    df = pd.DataFrame(i for i in data)
    #add the column heads
    df.columns= column_head
    df=df.sort_values(by=[fixed_params])
    #collect all dataframes in a dict just in case
    dataframe_collection[f"({fixed_params},{animal})"]=df

    #the actual plotting -- 
    ax.plot(10**df[fixed_param],df['loss']/optimal_loss[animal-1], 'go--',linewidth=1, markersize=3,c=colours[animal-1], label=f'animal {animal}')
    ax.plot(10**optim_params[animal-1][fixed_ind], 1, 'x', markersize=2, c='black')
    ax.hlines(3.71,10**df[fixed_param].iloc[0],10**df[fixed_param][-1:],ls='dashed',colors='grey',lw=0.5)
    ax.set_ylim(0,4.5)
ax.set_xlabel(labelling(fixed_param[0]), fontsize=15)
ax.set_ylabel('Norm. profile', fontsize=15)
#if fixed_param[0]=='t0':
#    ax.set_xscale('log')
ax.set_ylim(0,4)
ax.set_yticks([0,1,2,3])
#ax.set_xticks([0,0.1])
ax.xaxis.set_tick_params(labelsize=10)
#ax.annotate('a)',xy=(0,1.1))
ax.yaxis.set_tick_params(labelsize=10)
line,labels=ax.get_legend_handles_labels()
fig.legend(line,labels,loc='upper right', ncols=5, fontsize=8,bbox_to_anchor=(1,1.15))
fig.savefig(f"{folder}/testimage.png", bbox_inches='tight')

"""

fig2, ax2 = plt.subplots(figsize=(10,2.5),ncols=5, constrained_layout = True,sharey=True)

colordict={'$g_V$': (0.6509803921568628, 0.807843137254902, 0.8901960784313725), '$g_P$': (0.12156862745098039, 0.47058823529411764, 0.7058823529411765), '$k_R$': (0.6980392156862745, 0.8745098039215686, 0.5411764705882353), '$L$': (0.2, 0.6274509803921569, 0.17254901960784313),
            '$n_{\mathit{ves}}$': (0.984313725490196, 0.6039215686274509, 0.6), '$m_1$':  (0.9921568627450981, 0.7490196078431373, 0.43529411764705883), '$m_2$':(1.0, 0.4980392156862745, 0.0),'$N$': (0.792156862745098, 0.6980392156862745, 0.8392156862745098)}#dict(zip(new_param_Ids,sns.color_palette("Paired")[:len(new_param_Ids)]))
animal=4
fig10params = ['N_var','gV','gP','kR','nves']#
#fig10colors = {key:color[i] for i,key in enumerate(fig10params)}
#fig10params.remove('gP')
for i,p_fixed in enumerate(fig10params):
    parameter_names = ['N_var','gV','gP','kR','nves']#
    parameter_names.remove(p_fixed)

    #collecz the parameter values of animal 4 depending on which parameter was fixed in the calculations
    #(p_fixed) and norm them by the optimal parameter values of animal 4index 3)
    [ax2[i].plot(10**dataframe_collection[f"({p_fixed},{animal})"][p_fixed]/10**df_opt[p_fixed][animal-1],
                10**dataframe_collection[f"({p_fixed},{animal})"][key]/10**df_opt[key][animal-1], 
                'go-', label=f'{labelling(key)}', c=colordict[labelling(key)], lw=1, markersize=2)
                    for key in parameter_names]
    ax2[i].vlines(1,0,2,colors='grey',linestyle='dashed', lw=1)
    ax2[i].set_ylim(0,2)
    ax2[i].set_xlim(0,2)
    ax2[i].set_xticks([0,1,2])
    ax2[i].set_yticks([0,1,2])
    if p_fixed=='t0':
        ax2[i].set_xscale('log')
    ax2[i].xaxis.set_tick_params(labelsize=18)
    ax2[i].yaxis.set_tick_params(labelsize=18)
    ax2[i].set_xlabel(f'norm. {labelling(p_fixed)}', fontsize=18)
    parameter_names=parameter_names+[p_fixed]
ax2[0].set_ylabel('norm. parameter value', fontsize=18)
line,labels=ax2[0].get_legend_handles_labels()
line2,labels2=ax2[1].get_legend_handles_labels()
line = line+[line2[0]]
labels = labels +[labels2[0]]
for i in range(len(ax2.flatten())):
    ax2.flatten()[i].text(-0.15,1.2,letters[i],transform = ax2.flatten()[i].transAxes, fontsize=18)

fig2.legend(line,labels,loc='upper right', ncols=5, fontsize=18,bbox_to_anchor=(1,1.25))
    #plot fig 10 

fig2.savefig(f"{folder}/fig10objective.png", bbox_inches='tight')
