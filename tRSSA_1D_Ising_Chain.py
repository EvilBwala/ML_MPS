# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:05:31 2021

@author: Frank Gao
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from math import exp, log, cos, cosh, sqrt, sinh, pi

@jit(nopython=True)
def tRSSA_TD_1DChain(T, dmu, h, initial_chain=np.random.choice([-1,1], size=1000),
               P_gen_kj=np.ones((2,2))*0.5, J=4,
               Periods = 50, t_intervals = None, track_size = 10000):
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
    
    
    Periods: number of periods to simulate 
    
    there are 6 reactions
    1: -1 -> -1 1
    2: -1 -> -1 -1
    3: -1 -> 
    4: 1 -> 1 1
    5: 1 -> 1 -1
    6: 1 -> 
    
    Returns
    -------
    Timestamps and Changes of 1D-Chain
    '''     

    #Track the change of the 1D-Chain
    Chain_Changes = np.zeros(track_size)
    cc_i = 1000
    
    #track chain configuration
    initial_chain[-1] = -1
    initial_chain[-2] = 1
    chain = np.zeros(track_size)
    chain[0:1000] = initial_chain
    chain_length= len(chain)
    c_i = 1000
    
    #outmost blocks
    nj = initial_chain[-1]
    ni = initial_chain[-2]
    
    #track time
    reaction_time = np.zeros(track_size)
    rt_i = 0
    
    t_max = T*Periods
    #print(t_max)
    t_rn = 0
    t_next = t_intervals[1]
    dt = ts[1]

    #track time
    i = 1
        
    #print(t_intervals)    
    #print(t_next)
    
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
    #if h is positive, initial signal for boundary = 1 -1 is decreasing
    if h > 0:
        a_lb[2] = exp(-J*ni*nj-h*np.cos(2*pi*t_next/T)*nj)
        a_ub[2] = exp(-J*ni*nj-h*np.cos(2*pi*0/T)*nj)
    else:
        a_lb[2] = exp(-J*ni*nj-h*np.cos(2*pi*0/T)*nj) 
        a_ub[2] = exp(-J*ni*nj-h*np.cos(2*pi*t_next/T)*nj)
    a0_ub = np.sum(a_ub)
    
    
    while t_rn < t_max:
        
        #print(a_ub)
        
        tau = -1/a0_ub*log(np.random.random())
        t_rn = t_rn+tau
        #print(t_rn, t_next)
       
        
        if t_rn > t_next:
            t_rn = t_next
            i = i+1
            t_next = ts[i]
            
            #check if the signal is in first or second half
            first_half_T = True
            if np.floor(t_rn/(T/2))%2 == 1:
                first_half_T = False
            
            nj = chain[-1]
            ni = chain[-2]
            
            #update propensity bounds
            a_lb = np.zeros(6)
            a_ub = np.zeros(6)
            
            if nj == -1:
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
            
            if h > 0:
                if nj == -1:
                    #decresing signal for removal 
                    if first_half_T:
                        a_lb[2] = exp(-J*ni*nj-h*np.cos(2*pi*t_next/T)*nj)
                        a_ub[2] = exp(-J*ni*nj-h*np.cos(2*pi*t_rn/T)*nj)
                    #incresing signal
                    else:
                        a_lb[2] = exp(-J*ni*nj-h*np.cos(2*pi*t_rn/T)*nj)
                        a_ub[2] = exp(-J*ni*nj-h*np.cos(2*pi*t_next/T)*nj)
                else:
                    #increasing signal for removal
                    if first_half_T:
                        a_lb[5] = exp(-J*ni*nj-h*np.cos(2*pi*t_rn/T)*nj)
                        a_ub[5] = exp(-J*ni*nj-h*np.cos(2*pi*t_next/T)*nj)
                    #decreasing signal for removal
                    else:
                        a_lb[5] = exp(-J*ni*nj-h*np.cos(2*pi*t_next/T)*nj)
                        a_ub[5] = exp(-J*ni*nj-h*np.cos(2*pi*t_rn/T)*nj)
            else:
                if nj == -1:
                    #inscreasing signal for removal 
                    if first_half_T:
                        a_lb[2] = exp(-J*ni*nj-h*np.cos(2*pi*t_rn/T)*nj) 
                        a_ub[2] = exp(-J*ni*nj-h*np.cos(2*pi*t_next/T)*nj)
                    #incresing signal
                    else:
                        a_lb[2] = exp(-J*ni*nj-h*np.cos(2*pi*t_next/T)*nj)
                        a_ub[2] = exp(-J*ni*nj-h*np.cos(2*pi*t_rn/T)*nj)
                else:
                    #decreasing signal for removal
                    if first_half_T:
                        a_lb[5] = exp(-J*ni*nj-h*np.cos(2*pi*t_next/T)*nj) 
                        a_ub[5] = exp(-J*ni*nj-h*np.cos(2*pi*t_rn/T)*nj)
                    #decreasing signal for removal
                    else:
                        a_lb[5] = exp(-J*ni*nj-h*np.cos(2*pi*t_rn/T)*nj)
                        a_ub[5] = exp(-J*ni*nj-h*np.cos(2*pi*t_next/T)*nj)
            continue;
        
        
        #compare propensity to choose a reaction
        aj_ub_sums = np.zeros(6)
        aj_ub_sums[0] = a_ub[0]
        for j in range(1, 6):
            aj_ub_sums[j] = aj_ub_sums[j-1]+a_ub[j]
        r2 = np.random.random()
        r2a0_ub = r2*aj_ub_sums[-1]
        arg = 0
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
            if nj == -1:
                a0 = additon_a_matrix[0,1]+additon_a_matrix[0,0]+exp(-J*ni*nj-h*np.cos(2*pi*t_rn/T)*nj)
            else:
                a0 = additon_a_matrix[1,1]+additon_a_matrix[1,0]+exp(-J*ni*nj-h*np.cos(2*pi*t_rn/T)*nj)
            if r3 <= (a0/np.sum(a_ub)):
                Accept = True
        
        if Accept:      
            #choose a reaction
            #add +1
            if arg == 0 or arg == 3:
                chain_length += 1
                ni = nj
                nj = 1
                Chain_Changes[cc_i] = 1 
                chain[c_i] = 1
                c_i += 1
                cc_i += 1
                
                
            #add -1
            elif arg == 1 or arg == 4:
                chain_length += 1
                ni = nj
                nj = -1
                Chain_Changes[cc_i] = -1 
                chain[c_i] = -1
                c_i += 1
                cc_i += 1
            #removal
            else:
                chain_length -= 1
                c_i -= 1
                nj = chain[c_i]
                ni = chain[c_i-1]
                Chain_Changes[cc_i] = 2 
                cc_i += 1
                
            reaction_time[rt_i] = t_rn
            rt_i += 1
            
            if Chain_Changes[-1] != 0:
                Chain_Changes[0:1000] = Chain_Changes[-1000:]
                Chain_Changes[1000:] = np.zeros(track_size-1000)
                cc_i = 1000
            if reaction_time[-1] > 0:
                reaction_time[0:1000] = reaction_time[-1000:]
                reaction_time[1000:] = np.zeros(track_size-1000)
                rt_i = 1000
            if chain[-1] != 0:
                chain[0:1000] = chain[-1000:]
                chain[1000:] = np.zeros(track_size-1000)
                #print(t_rn, chain_length)
                c_i = 1000
            
            
    return chain_length, chain, Chain_Changes, reaction_time