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
    parser.add_argument("-x", nargs = 1, type = str, help = "path to the file where the internal states will be saved")
    parser.add_argument("-c", nargs = "?", type = str, help = "path to the file where the internal states will be saved")
    parser.add_argument("-e", nargs = "?", type = str, help = "path to the file where the absolute value of the eigen values will be saved")
    parser.add_argument("-p", nargs = "?", type = int, default = 0.95, help = "percentage of explanation")
    args = parser.parse_args(argv[1:])
    
    # Loading the internal states
    # --------------------------------------------------------------------------
    internals = np.load(args.x[0])

    # Building the components
    # --------------------------------------------------------------------------
    components = np.empty_like(internals)
    abs_eigs = np.empty((internals.shape[0], internals.shape[3]))
    for r in range(internals.shape[0]):
        internals_all = np.concatenate(internals[r], axis = 0)
        internals_all_m = np.mean(internals_all, axis = 0)
        internals_all_std = np.std(internals_all, axis = 0)
        internals_all_c = (internals_all - internals_all_m)
        internals_all_cr = internals_all_c/internals_all_std
        corr_internals_internals = np.dot(internals_all_cr.T, internals_all_cr)/(internals_all.shape[0])
        eig, v = np.linalg.eig(corr_internals_internals)
        abs_eig = np.abs(eig)
        idx = np.argsort(abs_eig)[::-1]
        sum_abs_eig = np.sum(abs_eig)
        p_pca = 0
        nb_principal_components = 0
        while p_pca < args.p*sum_abs_eig:
            p_pca += abs_eig[idx[nb_principal_components]]
            nb_principal_components += 1
        components_all = np.real(np.dot(internals_all_cr, v[:,idx]))
        components[r] = components_all.reshape(internals[r].shape)
        abs_eigs[r] = abs_eig
        print("In the reservoir {:d} there are {:d} principal components. They explain {:f}% of the data.".format(r,nb_principal_components,100*p_pca/sum_abs_eig))

    if args.c != None:
        np.save(args.c, components)
    if args.e != None:
        np.save(args.e, abs_eigs)

    
if __name__ == "__main__":
    main(sys.argv)

