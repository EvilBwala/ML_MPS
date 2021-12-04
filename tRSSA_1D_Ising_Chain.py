# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:05:31 2021

@author: Frank Gao
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from math import exp, log, cos, cosh, sqrt, sinh, pi
from time import process_time, time
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import pathlib


def last_n_nonzero(M, n):
    '''
    For a matrix M of m by N, 
    return a matrix of m by n by removing the zero at the end of each row and keep the last n element 
    '''
    m, N = M.shape
    New_M = np.zeros((m, n))
    for i in range(m):
        row_i = np.trim_zeros(M[i], 'b')
        New_M[i] = row_i[-n:]
    return New_M   


@jit(nopython=True)
def TD_1DChain(T, dmu, h, t_max = 50*1000, t_save_intervals = None,
               initial_chain=np.random.choice([2,1], size= 1000),
               P_gen_kj=np.ones((2,2))*0.5, J=4, seed = None,
               prop_t_bounds = None, track_size = 2000, keep_size = 1000):
    '''
    Time-dependent rejection-based SSA (tRSSA)
    
    Parameters
    ----------
    T: Period of the external cosine signal 
    dmu: The additional chemical potential from equilbirum 
    h:strength of the external signal
    
    P_gen_kj:  the probability of getting a particular particle nk from the bath given the interface particle is nj
       -1   1    
    -1 0.5 0.5
     1 0.5 0.5
    in general it is a matrix as above. Here just 0.5
    
    J: monomer-monomer interaction strength
    
    initial_chain_size: size for the initial bulk
    
    
    t_max: number of 
    to simulate 
    Burn_in_Periods: Statistics here will not be tracked
    
    prop_t_bounds: time for propensity bounds of the tRSSA algorithm
    
    
    
    there are 6 reactions
    1: -1 -> -1 1
    2: -1 -> -1 -1
    3: -1 -> 
    4: 1 -> 1 1
    5: 1 -> 1 -1
    6: 1 -> 
    
    Map the -1 to 2 for speed reasons
    
    Returns
    -------
    Timestamps and Changes of 1D-Chain
    '''     
    #np.random.seed(seed)
    
    #initial chain
    initial_chain[-1] = 2
    initial_chain[-2] = 1
    #outmost blocks
    nj = 2
    ni = 1
    
    #track chain configuration
    chain = np.zeros(track_size, dtype=np.uint8)
    chain[0:keep_size] = initial_chain.astype(np.uint8)
    chain_length= keep_size
    c_i = keep_size-1
    
    add_move = 0
    rem_move = 0
    
    
    #track time
    addition_reaction_time = np.zeros(track_size,dtype=np.float64)
    rt_i = 0
    
    t_rn = 0
    prop_t_next = prop_t_bounds[1]
    #track propensity time
    prop_t_i = 1
    #track save t_i
    save_t_i = 0
        
    
    #calculate chemical potential and rate constant for adding a spin
    mu_eq0 = log(2/(exp(J)*(cosh(h)+sqrt(exp(-4*J)+sinh(h)**2))))
    mu = mu_eq0+dmu
    exp_mu = exp(mu)
    
    #compute the time homogeneous rate for addition
    additon_a_matrix = exp_mu*P_gen_kj
    
    #bound for h is always 1, bound propoensity is same for rate fuction
    a_lb = np.zeros(6)
    a_ub = np.zeros(6)
    #add 1 to -1
    a1 = additon_a_matrix[0,1]
    a_lb[0] = a1
    a_ub[0] = a1
    #add -1 to -1
    a2 = additon_a_matrix[0,0]
    a_lb[1] = a2
    a_ub[1] = a2
    #removal of -1
    remove_cons = exp(J)
    signals = remove_cons*np.array([exp(h*cos(2*pi*prop_t_next/T)), exp(h*cos(2*pi*t_rn/T))])
    a_lb[2] = np.min(signals)
    a_ub[2] = np.max(signals) 
    a0_ub = np.sum(a_ub)
    
    
    #save signals, patterns, associated with the last n
    patterns_save = np.zeros((len(t_save_intervals), track_size), dtype=np.uint8)
    #signals_save = np.zeros((int(Periods-Burn_in_Periods), track_size))
    addition_reaction_time_save = np.zeros((len(t_save_intervals), track_size), dtype=np.float64)
    
    
    while t_rn < t_max:
        
        
        t_before = t_rn
        tau = -1/a0_ub*log(np.random.random())
        t_rn = t_rn+tau
        
        
        
        #check if we should save the patterns now
        if t_before < t_save_intervals[save_t_i] and t_rn > t_save_intervals[save_t_i]:
            patterns_save[save_t_i] = chain
            addition_reaction_time_save[save_t_i] = addition_reaction_time
            save_t_i += 1
        
        if t_rn > t_max:
            break;
        
        if t_rn > prop_t_next:
            t_rn = prop_t_next
            prop_t_i += 1 
            prop_t_next = prop_t_bounds[prop_t_i]
            
            nj = chain[c_i]
            ni = chain[c_i-1]
        
            #update propensity bounds
            a_lb = np.zeros(6)
            a_ub = np.zeros(6)
            
            if nj == 2:
                #add 1 to -1
                a1 = additon_a_matrix[0,1]
                a_lb[0] = a1
                a_ub[0] = a1
                #add -1 to -1
                a2 = additon_a_matrix[0,0]
                a_lb[1] = a2
                a_ub[1] = a2
            else:
                #add 1 to 1
                a4 = additon_a_matrix[1,1]
                a_lb[3] = a4
                a_ub[3] = a4
                #add -1 to 1
                a5 = additon_a_matrix[1,0]
                a_lb[4] = a5
                a_ub[4] = a5
            
            #compute removal propensity bounds
            if ni == 2 and nj == 2:
                remove_cons = exp(-J)
            elif ni == 1 and nj == 1:
                remove_cons = exp(-J)
            else:
                remove_cons = exp(J)
                
            if nj == 2:
                signals = remove_cons*np.array([exp(h*cos(2*pi*prop_t_next/T)), exp(h*cos(2*pi*t_rn/T))])
            else:
                signals = remove_cons*np.array([exp(-h*cos(2*pi*prop_t_next/T)), exp(-h*cos(2*pi*t_rn/T))])
                
            if nj == 2:
                a_lb[2] = np.min(signals)
                a_ub[2] = np.max(signals)   
            else:
                a_lb[5] = np.min(signals)
                a_ub[5] = np.max(signals)
            continue;
        
        nj = chain[c_i]
        ni = chain[c_i-1]
        #compare propensity to choose a reaction
        aj_ub_sums = np.zeros(6)
        aj_ub_sums[0] = a_ub[0]
        for j in range(1, 6):
            aj_ub_sums[j] = aj_ub_sums[j-1]+a_ub[j]
        r2 = np.random.random()
        r2a0_ub = r2*aj_ub_sums[-1]
        arg = 0
        #pick the reaction
        for j in range(6):
            if aj_ub_sums[j] > r2a0_ub:
                arg = j
                break;
        
        #check if the reaction should occur
        Accept = False
        r3 = np.random.random() 
        if r3 <= (np.sum(a_lb)/np.sum(a_ub)):
            Accept = True
        else:
            #evaluate propensity of current state
            a0 = 0
            if ni == 2 and nj == 2:
                remove_cons = exp(-J)
            elif ni == 1 and nj == 1:
                remove_cons = exp(-J)
            else:
                remove_cons = exp(J)
                
            if nj == 2:
                a0 = additon_a_matrix[0,1]+additon_a_matrix[0,0]+remove_cons*exp(+h*cos(2*pi*t_rn/T))
            else:
                a0 = additon_a_matrix[1,1]+additon_a_matrix[1,0]+remove_cons*exp(-h*cos(2*pi*t_rn/T))
            if r3 <= (a0/np.sum(a_ub)):
                Accept = True
        
        if Accept:      
            #choose a reaction
            #add +1
            if arg == 0 or arg == 3:
                #update chains
                chain_length += 1
                c_i += 1
                chain[c_i] = 1
                
                #update time
                addition_reaction_time[rt_i] = t_rn
                rt_i += 1
                
                add_move += 1
                
                
            #add -1
            elif arg == 1 or arg == 4:
                #update chains
                chain_length += 1
                c_i += 1
                chain[c_i] = 2
                
                #update time
                addition_reaction_time[rt_i] = t_rn
                rt_i += 1
                
                add_move += 1
                
            #removal
            else:
                chain_length -= 1
                #remove the spin from chain
                chain[c_i] = 0
                c_i -= 1
                nj = chain[c_i]
                ni = chain[c_i-1]
                
                rem_move += 1
                            
            if chain[-1] != 0:
                chain[:keep_size] = chain[-keep_size:]
                chain[keep_size:] = np.zeros(track_size-keep_size)
                c_i = keep_size-1
            if addition_reaction_time[-1] > 0:
                addition_reaction_time[0:keep_size] = addition_reaction_time[-keep_size:]
                addition_reaction_time[keep_size:] = np.zeros(track_size-keep_size)
                rt_i = keep_size
            
    #print(add_move, rem_move)
    return chain_length, patterns_save, addition_reaction_time_save
   
def tSRRA_1D_Chain_Process(T, h, t_max, t_burn_in, t_save_intervals, J=4, dmu=7, last_n = 20, save='', save_n = 0):
    prop_t_bounds = np.linspace(0, t_max, num=t_max*10+1)
    #t_save_intervals = np.linspace(t_burn_in, t_max, num=num_t_save_intervals)
    
    chain_length, patterns, addition_reaction_time = TD_1DChain(T, dmu, h, t_max = t_max,
                                                                        prop_t_bounds=prop_t_bounds,
                                                                        t_save_intervals = t_save_intervals)
        
    patterns_save = last_n_nonzero(patterns, last_n)
    addition_reaction_time_save = last_n_nonzero(addition_reaction_time, last_n)
    
    #compute average velocity (for the last n additions)
    avg_velocity_save = average_velocity(T, h, addition_reaction_time_save, t_save_intervals)
    
    
    vals_to_save = {'chain_length':chain_length,
                    'patterns': patterns_save, 
                    'addition_reaction_time': addition_reaction_time_save,
                    'average_velocity': avg_velocity_save}
    
    save_str = save[1:]+'\T='+str(T)+'_h='+str(h)+'_'+str(save_n)
    print(save_str)
    np.savez(save_str, **vals_to_save)
    
    
def Run_tSRRA_1D_Chain_Process(Ts, hs, t_max, num_save, trials = 100,
                               t_burn_in = 30*1000, J=4, dmu=7, 
                               last_n = 20, 
                               n_jobs = 1, save=''):
    
    
    #create the directory if it does not exist
    path = str(pathlib.Path().resolve())
    save_path = path+save
    isExist = os.path.exists(save_path)
    if not isExist:
        os.makedirs(save_path)
        
    t_save_intervals = np.linspace(t_burn_in, t_max, num=num_save)    
    
    for T in Ts:
        for h in hs:
            Parallel(n_jobs=n_jobs)(delayed(tSRRA_1D_Chain_Process)(T, h, t_max, t_burn_in, t_save_intervals, 
              J=J, dmu=dmu, last_n = last_n, save=save, save_n = i) for i in tqdm(range(trials)))     
            
    #compute an ensemble average velocity and the protocol
    for T in Ts:
        for h in hs:
            avg_vel_Th = np.zeros(num_save)
            for i in range(trials):
                avg_vel_Th_i = np.load(save_path+'\T='+str(T)+'_h='+str(h)+'_'+str(i)+'.npz')['average_velocity']
                avg_vel_Th = np.add(avg_vel_Th, avg_vel_Th_i)
            avg_vel_Th = avg_vel_Th/trials
            protocol, addition_time = protocol_from_average_velocity(T, h, avg_vel_Th, t_save_intervals, last_n)
            vals_to_save = {'protocol': protocol, 'addition_time':addition_time, 'average_velocity':avg_vel_Th}
            save_str = save[1:]+'\Protocol_T='+str(T)+'_h='+str(h)
            np.savez(save_str, **vals_to_save)
    

def average_velocity(T, h, addition_reaction_time, t_save_intervals):
    last_n = len(addition_reaction_time[0])
    num_t_save_intervals = len(t_save_intervals)
    avg_vel = np.zeros(num_t_save_intervals)
    for i in range(num_t_save_intervals):
        #velocity = number of blcoks/time interval
        avg_vel[i] = last_n/(addition_reaction_time[i][-1]-addition_reaction_time[i][0])
    return avg_vel

def protocol_from_average_velocity(T, h, average_velocity, t_save_intervals, last_n):
    '''
    Using the ensemble (T, h) average average-velocity to estimate signal 
    '''
    num_t_save_intervals = len(t_save_intervals)
    protocol = np.zeros( (num_t_save_intervals, last_n))
    approximate_addition_time = np.zeros( (num_t_save_intervals, last_n))
    for i in range(num_t_save_intervals):
        inv_avg_vel_i =  1/average_velocity[i]
        #compute the approximated addition time using average velocity
        approximate_addition_time[i, -1] = t_save_intervals[i]
        for j in range(2, last_n+1):
            approximate_addition_time[i, -j] = approximate_addition_time[i, -j+1] - inv_avg_vel_i
        protocol = h*np.cos(2*pi/T*approximate_addition_time)
            
    return protocol, approximate_addition_time
        

def bits_patterns(n):
    '''
    Generate all patterns of 1&-1's of size n
    '''
    ps = ['-1', '1']
    for i in range(n-1):
        ps_temp = []
        for i in range(len(ps)):
            ps_temp.append(ps[i]+'-1')
            ps_temp.append(ps[i]+'1')
        ps = ps_temp
    return ps  

def bits_pattern_dict(n):
    two_to_n = 2**n
    n_bits_patterns = bits_patterns(n)
    #create a dictionary to store patterns index
    n_bits_patterns_dict = {}
    for i in range(two_to_n):
        n_bits_patterns_dict[n_bits_patterns[i]] = i
    return n_bits_patterns_dict
    
    
    
                              
