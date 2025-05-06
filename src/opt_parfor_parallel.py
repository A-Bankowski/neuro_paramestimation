import time
from matplotlib import pyplot as plt
from multiprocessing import Pool
from scipy.optimize import minimize
from numpy import exp

import importlib
import os
import sys
import libsbml
#sys.path.insert(0, '/data/numerik/people/abankowski/neuro_param_estimation/codes')
import amici
import amici.plotting
import numpy as np

import pypesto
import pypesto.optimize as optimize
import pypesto.visualize as visualize
from scipy.signal import find_peaks
from scipy import signal
from numpy import log10
from cycler import cycler
from mpi4py import MPI

COMM = MPI.COMM_WORLD

from opt_settings import *
from embarrassingly_parallel_forloop import *

plt.rcParams.update({
    "font.family": "serif",
   "font.serif": "cm",
    "axes.prop_cycle": cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', 
            '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']),
    })
# -- somehow doesn't work on computing cluster
###############################################################################################################################################################################################
"""
This version uses @parFor. In opt_settings set the number of starts. When submitting the job file, choose an appropriate number of nodes to distribute the tasks.
"""
###############################################################################################################################################################################################



# name of SBML file
sbml_file = "recovery_model_60_peaks_qssa_rates.xml"
# name of the model that will also be the name of the python module
model_name = "recovery_model_60_peaks_qssa_rates" 
#directory to which the generated model code is written
model_output_dir = "recovery_model_60_peaks_qssa_rates"#model_name


"""
If this is the first time running the code and there is no recovery_model_60_peaks_qssa_rates python module, run this to import it with amici 
"""

#sbml_reader = libsbml.SBMLReader() #blub
#sbml_doc = sbml_reader.readSBML(sbml_file)
#sbml_model = sbml_doc.getModel()
#import sbml model, compile and generate amici module
#sbml_importer = amici.SbmlImporter(sbml_file)
#sbml_importer.sbml2amici(model_name, model_output_dir, verbose=True,generate_sensitivity_code = False)



# load amici module (the usual starting point later for the analysis) os.path.abspath(model_output_dir)
sys.path.insert(0, model_output_dir)
model_module =  amici.import_model_module(model_name,model_output_dir)
model = model_module.getModel()
ts = np.load('/data/numerik/people/abankowski/neuro_param_estimation/codes/ts.npy')
times = ts
np.save('/data/numerik/people/abankowski/neuro_param_estimation/codes/times.npy',times)

""" 
loading the times and the solver 
"""

times = np.load('/data/numerik/people/abankowski/neuro_param_estimation/codes/times.npy')
timestep=np.diff(times)[0]
#print(ts,times,ts.shape,times.shape)
model.setTimepoints(times)
model.setParameterScale(amici.ParameterScaling.log10)
print('tstep=',timestep)
solver = model.getSolver()
solver.setNewtonMaxSteps(100)
solver.setMaxSteps(1599999)

"""
Get the raw data as well as the parameter names
"""
# how to run amici now:
rdata = amici.runAmiciSimulation(model, solver,None)
amici.plotting.plotStateTrajectories(rdata)
#plt.savefig('nominal_plot')

old_param_names = list(model.getParameterIds())
new_param_names = deepcopy(old_param_names)
old_not_ = ['N_var','gV','gP','kR','t0','k_base','fac','L','nves','nsites']
new_not_ = ['$N$', '$g_V$', '$g_P$', '$k_R$', '$t_0$', '$k_0$', '$\gamma$', '$L$', '$n_{\\rm{ves}}$', '$n_{\\rm{sites}}$' ]
for i in range(len(old_not_)):
    id_= old_param_names.index(old_not_[i])
    new_param_names[id_] = new_not_[i]

# print model information
print("Parameter values",model.getParameters())
print("Model name:", model.getName())
print("Model parameters:", model.getParameterIds())
print("Model outputs:   ", model.getObservableIds())
print("Model states:    ", model.getStateIds())

# initialize the variables
variables = { name:10**np.array(model.getParameters())[i] for i,name in enumerate(model.getParameterIds()) }

#für parameterschätzung III
if param_est ==3:
    variables['L'] = 1.6
    variables['k_base'] = 18.7
    variables['t0'] = 3.9e-4

#???
t = amici.runAmiciSimulation(model, solver, None).t
R = amici.runAmiciSimulation(model, solver, None).x[:,0]

for n, val in enumerate([20 for i in range(50)]):
    n+=61
    globals()["mu%d"%n] = val
    
if (measurements=='real'):
	variables['t_wait'] = variables['t_wait'] - 0.0014*1.9
nsites = number_of_sites
model.setParameters(np.log10(np.array([variables[key] for key in variables])))

#used in the params_0 intitialisation
nom = model.getParameters()

"""
The observed fusions multiplied by factor N. (N*\dot{F}). First amici solves the ODE given the parameters p. Then k_F(t) is calculated given the analytic representation.
Input: Parameters p, np array
Output: N*\dot{F} = N*k_F(t)*R(t)
"""
def NF(p):
	model.setParameters(p)
	t = amici.runAmiciSimulation(model, solver, None).t
	R = amici.runAmiciSimulation(model, solver, None).x[:,0]
	variables = {key:10**p[i] for i,key in enumerate(model.getParameterIds())}
	baseline = 10**(variables['L']*(1-exp(-variables['k_base']*((t*variables['fac']-variables['t_wait'])-variables['t0']))))
	peaks = np.sum([variables[f'a{i}']*exp(-0.5*((t*variables['fac']-variables['t_wait'])-variables[f'mu{i}'])**2/variables['sigma']**2) for i in range(1,61)],axis=0)
	kF = baseline + peaks
	return variables['N_var']*kF*R

"""
Define the mEPSC function. 
Input: tstep
Output: mEPSC mini current used for the convolution with \dot{F}k_F(t) to create the measured current
"""

def mEPSC_fun(tstep):
    ###Parameters, don't change!
    size_of_mini = 0.6e-9 #A, Amplitude of mEJC, Estimated from variance-mean of data (see Fig 2F)
    A = -7.209251536449789e-06
    B = 2.709256850482493e-09
    t_0 = 0
    tau_rf = 10.692783377261414
    tau_df =0.001500129264510
    tau_ds = 0.002823055510748#*0.6
    length_of_mini =34*1e-3
    
    """Return one mEPSC."""
    t = np.arange(0,length_of_mini,tstep)
    mEPSC = (t >= t_0)*(A*(1-np.exp(-(t-t_0)/tau_rf))*(B*np.exp(-(t-t_0)/tau_df) + (1-B)*np.exp(-(t-t_0)/tau_ds)))
    mEPSC = -(mEPSC/min(mEPSC) *size_of_mini)
    
    return mEPSC

"""
Calculate the resulting current by convolving with the impulse function mEPSC.
Input: Fusions NF =N*\dot{F}
Output: Current = N\dot{F}*mESPC (convolution)
"""
def current(NF):
    return signal.convolve(NF*timestep, mEPSC_fun(timestep))

"""
Objective function for the optimisation process. The peaks are used as reference. 
"""

t = amici.runAmiciSimulation(model, solver, None).t

#path strings for loading data
current_data = 'data/Current_data_animal'+str(animal)+'.npy'
nf_data = 'data/NF_data_animal'+str(animal)+'.npy'

stop_ind=[-9,-4,-5,-7,-9]

#find the peaks in the real data to compare with the simulated ones
if (measurements == 'real'):
    NF_data = np.load(nf_data)
    current_data=np.load(current_data)
    h=20000   # what is that?
    #
    top_peaks_data= find_peaks(current_data,distance=10,height=(-3.9e-8,-0.4e-8))[0][:stop_ind[animal-1]]
    bottom_peaks_data = find_peaks(-current_data,3e-8)[0]
    NF_peaks_data = find_peaks(NF_data,height=h)[0]

#definition of the objective function
                              
if (objective == 'peaks_cur'):
    def f_obj(param):
        #get the current parameters
        param_Id = list(model.getParameterIds()[i] for i in param_indices)
        model.setParameterById(dict(zip(param_Id, param)))
        all_params = np.array(model.getParameters())

        #calculate the current
        R = amici.runAmiciSimulation(model, solver).x[:,0]
        t = amici.runAmiciSimulation(model, solver).t
        NF_model = NF(all_params)
        NF_peaks_model = find_peaks(NF_model)[0]
        current_model = current(NF_model)
        
        #find the peaks in the simulated data
        top_peaks_model = find_peaks(current_model)[0]
        
        bottom_peaks_model = find_peaks(-current_model)[0]

        if (59 <= len(current_model[top_peaks_model])) & (60 <= len(current_model[bottom_peaks_model])):
            #return the difference at the peaks (top and bottom) multiplied by factor 10⁸
            fehler = ((current_data[top_peaks_data]*10**9 - current_model[top_peaks_model][:59]*10**9)**2).sum() + ((current_data[bottom_peaks_data]*10**9 - current_model[bottom_peaks_model][:60]*10**9)**2).sum()
            #print(fehler)
            return fehler
        else: 
            print("Inf")
            return np.inf

#    
# definition of optimization function
def optimize_params(param_0):
    print("SETTING",list(model.getParameterIds()[i] for i in param_indices))
    print("to",param_0)
    print("with bounds",param_bounds)
    opt_ = minimize(f_obj, param_0, method = opt_algotrithm, bounds=param_bounds, tol=1e-3, options=opt_options)
    print(opt_)
    opt_res = opt_.x
    return opt_res

"""
Initialize the parameters
"""
params_0_list=[]
#used in the params_0 intitialisation
nom = model.getParameters()
for i in range(len(param_indices)):
    # create the bounds of the initial radius
    nom_val = 10**nom[param_indices[i]]
    lb_log,ub_log = param_bounds[i]
    lb=10**lb_log
    ub=10**ub_log
    
    lb_radius=(1-initial_radius_factor)*nom_val
    ub_radius=(1+initial_radius_factor)*nom_val
    
    if lb_radius < lb or ub_radius > ub:

        raise Exception("initial_radius_factor outside allowed bounds",model.getParameterIds()[param_indices[i]],lb,lb_radius,ub,ub_radius,nom_val)
    
    else:
        random_betweens=np.random.default_rng(seed=seed).uniform(lb_radius,ub_radius,n_starts)
        params_0_list.append(np.log10(random_betweens))
params_0=np.array(params_0_list).T
#params_0 = np.array([np.load(f"{exp_dir}")[ind]])
np.save(f'{run_dir}/initial_params_array.npy',params_0)

@parFor(params_0,COMM)
def optimize_wrapper(param,COMM):
    return optimize_params(param)
#%%
#parallel optimization for all initial values.
if __name__ == "__main__":

    #embarrassingly parallel for-loop --- give multiple starting points (n_starts many) within params_0 and start optimizing for each in parallel
    start_time = time.perf_counter()
    result = optimize_wrapper(None, COMM)
    if COMM.rank == 0:
        print("Optimized Results:", 10**result[0])
    finish_time = time.perf_counter()

    print("Program finished in {} seconds - using embarrassingly parallel for-loops".format(finish_time-start_time))
    print("---")
    print("Result (in 10**)",10**result[0])
    #save the parameter values
    optimized_params =np.array(result)
    np.save(f'{run_dir}/optimized_params_array.npy',optimized_params)

# %%
