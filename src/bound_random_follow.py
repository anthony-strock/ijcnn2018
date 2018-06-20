#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - Xavier Hinaut, Nicolas P. Rougier, Anthony Strock
# Released under the BSD license
# ------------------------------------------------------------------------------
# This is an implementation of a reservoir computing model that learns to
# constantly output the input value at the moment of an arbitrary trigger.
#
#        ┌────────────────────────────────────────────────┐
#        │    ┌───────────┐                               │
#        └──▶︎ │           │               ┌──────────┐    │
#   input ──▶︎ │           │  	 bias ──▶︎ │          │    │
# trigger ──▶︎ │ reservoir │─────────────▶︎ │ readout  │────┴───▶︎ output
#    bias ──▶︎ │           │               │          │
#             │           │               └──────────┘
#             └───────────┘
#
# Example:                tick
#                          ↓
#   input:   4 1 5 7 2 3 9 4 7 6 3 1 1 8 ...
#   tick:    0 0 0 0 0 0 0 1 0 0 0 0 0 0 ...
#   output:  0 0 0 0 0 0 0 4 4 4 4 4 4 4 ...
#
# In this experiment, we...
# ------------------------------------------------------------------------------

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import trange

def set_seed(seed = None, old_rstate = None, verbose = 0):
    if seed == None:
        seed = int(time.time())%4294967296
    if verbose > 0:
        print("Seed:", seed)
    if old_rstate == None:
        return seed, np.random.RandomState(seed)
    else:
        old_rstate.seed(seed)
        return seed, old_rstate

def main(argv):
    parser = argparse.ArgumentParser(description=argv[0]+': Evaluation of a task')
    parser.add_argument("seed", nargs = "?", type = int, help = "seed used", default = None)
    parser.add_argument("-o", nargs = 1, type = str, help = "path to the file where the absolute erro will be saved")
    args = parser.parse_args(argv[1:])
    # Parameters
    # --------------------------------------------------------------------------
    seed = args.seed # Predefined seed, if None one will be chosen
    seed, rstate = set_seed(seed = seed, verbose = 1) # Seed used
    n_res = 100 # Number of reservoir tested
        # Training parameters
        # ----------------------------------------------------------------------
    n_samples_train = 100       # Number of samples
    sample_size_train = 100     # Sample temporal size
    n_trigger_train = 5         # Number of trigger in each sample
    trigger_earliest_train = 30 # Earliest allowed time for trigger (warmup)
    trigger_latest_train = 50   # Latest allowed time for trigger
        # Test parameters
        # ----------------------------------------------------------------------
    n_samples_test = 100     # Number of samples
    sample_size_test = 10000 # Sample temporal size
    trigger_time_test = 100  # Time when the first trigger occurs (to align errors)
        # Network hyper parameters
        # ----------------------------------------------------------------------
    n_unit = 100  # Number of unit in the reservoir
    n_input = 2   # Number of input
    n_output = 1  # Number of output
    leak = 8.76408736338339e-08  # Leak rate
    radius = 1.1031930545457658e-09  # Spectral radius
    s_input = 0.9308322888326782 # Input scaling
    s_fb = 0.3267791497571466    # Feedback scaling
    ridge = 5.214966493068512e-11 # Regularization coefficient in the ridge regression
    
    # Build weights
    # --------------------------------------------------------------------------
    res_seeds = rstate.randint(4294967296, size = (n_res,))
    W_in = np.empty((n_res, n_unit,n_input))
    W_fb = np.empty((n_res, n_unit,n_output))
    W = np.empty((n_res, n_unit, n_unit))
    C_in = np.empty((n_res, n_unit))
    for r in range(n_res):
        res_seed, res_rstate = set_seed(res_seeds[r])
        # Input weight matrix
        W_in[r] = (res_rstate.uniform(size = (n_unit,n_input)) - 0.5)*(2*s_input)
            # Feedback weight matrix
        W_fb[r] = (res_rstate.uniform(size = (n_unit,n_output)) - 0.5)*(2*s_fb)
            # Recurrent weight matrix
        W[r] = res_rstate.uniform(size = (n_unit, n_unit))-0.5
        actual_radius = max(abs(np.linalg.eig(W[r])[0]))
        if actual_radius > 0.:
            W[r] *= radius/actual_radius
        else:
            raise NameError("Null spectral radius")
        # Input bias matrix
        C_in[r] = (res_rstate.uniform(size = (n_unit))-0.5)*s_input
    W_out = np.empty((n_res, n_output, n_unit))
    C_out = np.empty((n_res, n_output))
    
    # Build Training Samples
    # --------------------------------------------------------------------------
        # Inputs
    inputs = np.empty( (n_samples_train, sample_size_train, n_input) )
            # Data to store
    inputs[:,:,0] = rstate.uniform(low = -1., high = 1., size = (n_samples_train, sample_size_train))
            # Triggers
    inputs[:,:,1] = 0
    trigger_times = np.empty((n_samples_train, n_trigger_train), dtype = np.int64)
    for k in range(n_samples_train):
        trigger_times[k] = np.sort(rstate.permutation(np.arange(trigger_earliest_train, trigger_latest_train))[:n_trigger_train])
        inputs[k,trigger_times[k],1] = 1
        # Desired outputs
    desired_outputs = np.zeros( (n_samples_train, sample_size_train, n_output) )       
    for k in range(n_samples_train):
        for l in range(n_trigger_train):
            desired_outputs[k,trigger_times[k,l]:] = inputs[k,trigger_times[k,l],0]
            
    # Offline learning
    # --------------------------------------------------------------------------
    n_training_points = (sample_size_train - trigger_earliest_train) * n_samples_train
    X = np.empty((1 + n_unit, n_training_points))
    Y = np.empty((n_output, n_training_points))
    internal = np.empty((sample_size_train, n_unit))
    for r in range(n_res):
        index = 0
        for i in range(inputs.shape[0]):
            input_ = inputs[i]
            output = desired_outputs[i]
            internal[0] = leak*np.tanh(np.dot(W_in[r],input_[0]) + C_in[r])
            for t in range(1,input_.shape[0]):
                internal[t] = np.tanh(np.dot(W[r],internal[t-1]) + np.dot(W_in[r],input_[t]) + np.dot(W_fb[r],output[t-1]) + C_in[r])
                internal[t]= leak*internal[t]+(1-leak)*internal[t-1]                
                if t>=trigger_earliest_train:
                    X[:, index] = np.concatenate([[1.0], internal[t]])
                    Y[:, index] = output[t]             
                    index += 1
        assert(index == n_training_points)
        A = np.dot(Y, np.dot(X.T, np.linalg.inv(np.dot(X,X.T) + (ridge**2)*np.identity(X.shape[0]))))
        C_out[r] = A[:, 0]
        W_out[r] = A[:, 1:]

    # Build Testing Samples
    # --------------------------------------------------------------------------
        # Inputs
    inputs = np.empty((n_samples_test, sample_size_test, n_input))
            # Data to store
    inputs[:,:,0] = rstate.uniform(low = -1., high = 1., size = (n_samples_test, sample_size_test))
            # Triggers
    inputs[:,:,1] = 0
    for k in range(n_samples_test):
        for t in range(trigger_time_test, sample_size_test):
            inputs[k,t,1] = 1
        # Outputs
    desired_outputs = np.zeros((n_samples_test, sample_size_test, n_output))       
    for k in range(n_samples_test):
        for t in range(trigger_time_test, sample_size_test):
            desired_outputs[k,t] = inputs[k,t,0]

    # Testing
    # --------------------------------------------------------------------------
    internals = np.empty((n_samples_test, sample_size_test, n_unit))
    outputs = np.empty((n_samples_test, sample_size_test, n_output))
    abs_error = np.empty((n_res, n_samples_test, sample_size_test, n_output))
    print("{:s} will approximately take {:f} GB".format(args.o[0], (n_res*n_samples_test*sample_size_test*n_output*4)/(2**30)))
    for r in trange(n_res, desc = "Testing reservoirs"):
        for i in trange(n_samples_test, desc = "Samples tested"):
            internals[i,0] = leak*np.tanh(np.dot(W_in[r], inputs[i,0]) + C_in[r])
            outputs[i,0] = np.clip(np.dot(W_out[r], internals[i,0]) + C_out[r],-1,1)  
            for n in range(1, sample_size_test):
                internals[i,n] = np.tanh(np.dot(W[r], internals[i,n-1]) + np.dot(W_fb[r], outputs[i,n-1]) + np.dot(W_in[r], inputs[i,n]) + C_in[r])
                internals[i,n] = (1-leak)*internals[i,n-1] + leak*internals[i,n]
                outputs[i,n] = np.clip(np.dot(W_out[r], internals[i,n]) + C_out[r],-1,1)
        abs_error[r] = np.abs(outputs - desired_outputs)
    
    np.save(args.o[0], abs_error)
    
    rms = np.sqrt(np.sum(abs_error**2, axis = (2,3))/(abs_error.shape[2]))
    
    mean_rms = np.mean(rms)
    std_rms = np.std(rms)
    
    print("Error: {:e} ± {:e}".format(mean_rms, std_rms))
    
if __name__ == "__main__":
    main(sys.argv)

