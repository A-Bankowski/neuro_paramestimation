# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:24:59 2020

@author: Arbeit
"""

from mpi4py import MPI
import functools
import time
import numpy as np


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        if MPI.COMM_WORLD.rank ==0:
            start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        
        if MPI.COMM_WORLD.rank ==0:
            end_time = time.perf_counter()      # 2
            run_time = end_time - start_time    # 3
            print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer

def split(container, count):
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]


# def parFor(data, COMM,verbose=False):
#     """A decorator that can be applied to an operation that should be executed independently but in the same
#      way on an array of data (embarassingly parallel for-loop). For example, say you want to add 1 to all numbers 
#      in data = [1,2,3,4]. Since these operations are independent, you can execute them in parallel without giving it
#      much thought. To do so, you apply @parFor(data,COMM) to your looped operation, (in this case something like
#     def myloop(d, COMM): return d+1. The d you write when calling the decorated function will be overwritten (by d in data) and 
#     is just a dummy."""
#     results = []
#     def middle(func):
#         def wrapper(*args, **kwargs):
#             nonlocal results
#             i=0
#             # Collect whatever has to be done in a list. Here we'll just collect a list of
#             # numbers. Only the first rank has to do this.
#             if COMM.rank == 0:
#                 jobs = list(range(len(data)))
#                 # Split into however many cores are available.
#                 jobs = split(jobs, COMM.size)
                
#             else:
#                 jobs = None
            
#             # Scatter jobs across cores.
#             print(jobs)
#             jobs = COMM.scatter(jobs, root=0)
            
            
#             # Now each rank just does its jobs and collects everything in a results list.
#             # Make sure to not use super big objects in there as they will be pickled to be
#             # exchanged over MPI.

            
            
#             for job in jobs:
#                 #if verbose: print("job %d/%d"%(job+1,len(data)))
#                 d = data[job]  
#                 jobresult = func(d, COMM)
#                 results.append(jobresult)
#                 i+=1
            
#             # Gather results on rank 0.
            
#             results = MPI.COMM_WORLD.gather(results, root=0)
            
            
#             if COMM.rank == 0:
#                 # Flatten list of lists.
#                 results = [_i for temp in results for _i in temp]

#             # make sure every rank has the same result
#             results = COMM.bcast(results,root=0)        
#             return results
#         return wrapper 
#     return middle

def parFor(data, COMM,verbose=False):
    """A decorator that can be applied to an operation that should be executed independently but in the same
     way on an array of data (embarassingly parallel for-loop). For example, say you want to add 1 to all numbers 
     in data = [1,2,3,4]. Since these operations are independent, you can execute them in parallel without giving it
     much thought. To do so, you apply @parFor(data,COMM) to your looped operation, (in this case something like
    def myloop(d, COMM): return d+1. The d you write when calling the decorated function will be overwritten (by d in data) and 
    is just a dummy."""
    results = []
    order =  []
    def middle(func):
        def wrapper(*args, **kwargs):
            nonlocal results
            nonlocal order
            i=0
            # Collect whatever has to be done in a list. Here we'll just collect a list of
            # numbers. Only the first rank has to do this.
            if COMM.rank == 0:
                jobs = list(range(len(data)))
                # Split into however many cores are available.
                jobs = split(jobs, COMM.size)
                
                joblengths = np.asarray([len(j) for j in jobs])
                joblengths_unique = np.unique(joblengths)
                numbers = np.zeros_like(joblengths_unique)
                for j in joblengths:
                    n_index = np.where(joblengths_unique ==j)
                    numbers[n_index]+=1
                print("numbers:",numbers,"joblenghts:",joblengths_unique)
                
            else:
                jobs = None
            
            # Scatter jobs across cores.
            #print(jobs)
            jobs = COMM.scatter(jobs, root=0)
            #print(jobs)
            
            # Now each rank just does its jobs and collects everything in a results list.
            # Make sure to not use super big objects in there as they will be pickled to be
            # exchanged over MPI.

            
            
            for job in jobs:
                #if verbose: print("job %d/%d"%(job+1,len(data)))
                d = data[job]  
                jobresult = func(d, COMM)
                results.append(jobresult)
                order.append(job)
                i+=1
                #print("HERE",COMM.rank, job, jobresult, d)
            
            # Gather results on rank 0.
            
            
            
            results = COMM.gather(results, root=0)
            order = COMM.gather(order, root=0)
            
            if COMM.rank == 0 and results!=None:
                # Flatten list of lists.
                #print("inside deco", results, "size",COMM.Get_size(), "rank",COMM.Get_rank(), "data",data, "d",d)
                results = [_i for temp in results for _i in temp]
                order = [_i for temp in order for _i in temp]
                
                results =[x for _, x in sorted(zip(order, results))]
                

            # make sure every rank has the same result
            results = COMM.bcast(results,root=0)  
            order = COMM.bcast(order,root=0)
            return results#,order
        return wrapper 
    return middle


def assign_color(rank,worldsize,splitnr):
    a=np.arange(worldsize)
    splitt = split(a,splitnr)
    
    for chunknr in range(len(splitt)):
        chunk = splitt[chunknr]
        if rank in chunk:
            color= chunknr
    return color

# def parFor2(data, COMM,verbose=False):
#     """A decorator that can be applied to an operation that should be executed independently but in the same
#      way on an array of data (embarassingly parallel for-loop). For example, say you want to add 1 to all numbers 
#      in data = [1,2,3,4]. Since these operations are independent, you can execute them in parallel without giving it
#      much thought. To do so, you apply @parFor(data,COMM) to your looped operation, (in this case something like
#     def myloop(d, COMM): return d+1. The d you write when calling the decorated function will be overwritten (by d in data) and 
#     is just a dummy."""
    
#     results = []
#     def middle(func):        
#         def wrapper(*args, **kwargs):
#             print("RANK",COMM.Get_rank())
#             nonlocal results
#             i=0
#             # Collect whatever has to be done in a list. Here we'll just collect a list of
#             # numbers. Only the first rank has to do this.
#             print("RANK",COMM.Get_rank())
#             if COMM.rank == 0:
#                 jobs = list(range(len(data)))
#                 # Split into however many cores are available.
#                 jobs = split(jobs, COMM.size)
                
#             else:
#                 jobs = None
            
#             # Scatter jobs across cores.
            
#             jobs = COMM.scatter(jobs, root=0)
            
            
#             # Now each rank just does its jobs and collects everything in a results list.
#             # Make sure to not use super big objects in there as they will be pickled to be
#             # exchanged over MPI.

            
            
#             for job in jobs:
#                 #if verbose: print("job %d/%d"%(job+1,len(data)))
#                 d = data[job]  
#                 jobresult = func(d, COMM)
#                 results.append(jobresult)
#                 i+=1
            
#             # Gather results on rank 0.
            
#             results = MPI.COMM_WORLD.gather(results, root=0)
            
            
#             if COMM.rank == 0:
#                 # Flatten list of lists.
#                 results = [_i for temp in results for _i in temp]

#             # make sure every rank has the same result
#             results = COMM.bcast(results,root=0)        
#             return results
#         return wrapper 
#     return middle


