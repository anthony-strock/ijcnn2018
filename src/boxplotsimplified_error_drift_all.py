#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - Xavier Hinaut, Nicolas P. Rougier, Anthony Strock
# Released under the BSD license
# ------------------------------------------------------------------------------
# ...
# ------------------------------------------------------------------------------

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

def main(argv):
    parser = argparse.ArgumentParser(description=argv[0]+': Evaluation of a task')
    parser.add_argument("once", nargs = 1, type = str, help = "path to the file where the absolute error is saved for the once condition")
    parser.add_argument("periodic", nargs = 1, type = str, help = "path to the file where the absolute error is saved for the periodic condition")
    parser.add_argument("follow", nargs = 1, type = str, help = "path to the file where the absolute error is saved for the follow condition")
    parser.add_argument("-o", nargs = 1, type = str, help = "path to the file where the figure will be saved")
    args = parser.parse_args(argv[1:])
    # Parameters
    # --------------------------------------------------------------------------
    trigger_time = 100
    trigger_step = 1000
   
    percentiles = [5,50,95]
    linestyles = [":","--","-","--",":"]
    max_value = 1e0
    min_value = 1e-6
    
    # Loading data
    # --------------------------------------------------------------------------    
    abs_error = np.load(args.once[0])
    print(abs_error.shape)
    percentiles_abs_error_once = np.percentile(abs_error, percentiles, axis = (0,1))
    print(percentiles_abs_error_once.shape)
    
    
    abs_error = np.load(args.periodic[0])
    print(abs_error.shape)
    percentiles_abs_error_periodic = np.percentile(abs_error, percentiles, axis = (0,1))
    
    abs_error = np.load(args.follow[0])
    percentiles_abs_error_follow = np.percentile(abs_error, percentiles, axis = (0,1))
    
    sample_size = abs_error.shape[2]
    
    # Display Once
    # --------------------------------------------------------------------------
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    plt.rc("ytick", labelsize="x-small")
    plt.rc("xtick", labelsize="x-small")
    
    f = plt.figure(figsize=(5,5))
    
    ax = f.add_subplot(3,1,1)
    X = np.arange(sample_size)
    ax.axvline(trigger_time, 0, 1, color="red", linestyle = "--", linewidth = 0.5)
    ax.fill_between(X, percentiles_abs_error_once[0,:,0], percentiles_abs_error_once[2,:,0], facecolor = "black", edgecolor = None, alpha = 0.1)
    ax.plot(X, percentiles_abs_error_once[1,:,0], linestyle = "-", linewidth=0.5, color = "black")
    ax.text(trigger_time+120, max_value, "▼",
            va="bottom", ha="right", fontsize="small", color = "red",
            bbox={'facecolor' : 'white', 'edgecolor' : 'none',
                  'pad': 0.5, 'alpha': 0.0})
    ax.set_ylim(min_value, max_value)
    ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Error")
    ax.text(sample_size+500, 1e-3, "A")
    
    ax = f.add_subplot(3,1,2)
    X = np.arange(sample_size)
    for t in range(trigger_time, sample_size, trigger_step):
        ax.axvline(t, 0, 1, color="red", linestyle = "--", linewidth = 0.5)
        ax.text(t+120,  max_value, "▼",
            va="bottom", ha="right", fontsize="small", color = "red",
            bbox={'facecolor' : 'white', 'edgecolor' : 'none',
                  'pad': 0.5, 'alpha': 0.0})
    ax.fill_between(X, percentiles_abs_error_periodic[0,:,0], percentiles_abs_error_periodic[2,:,0], facecolor = "black", edgecolor = None, alpha = 0.1)
    ax.plot(X, percentiles_abs_error_periodic[1,:,0], linestyle = "-", linewidth=0.5, color = "black")
    ax.set_ylim(min_value, max_value)
    ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Error")
    ax.text(sample_size+500, 1e-3, "B")
    
    ax = f.add_subplot(3,1,3)
    X = np.arange(sample_size)
    ax.axvline(trigger_time, 0, 1, color="red", linestyle = "--", linewidth = 0.5)
    ax.hlines(max_value, trigger_time, sample_size, color="red", linestyle = ":", linewidth = 0.5)
    ax.fill_between(X, percentiles_abs_error_follow[0,:,0], percentiles_abs_error_follow[2,:,0], facecolor = "black", edgecolor = None, alpha = 0.1)
    ax.plot(X, percentiles_abs_error_follow[1,:,0], linestyle = "-", linewidth=0.5, color = "black")
    ax.text(trigger_time+120, max_value, "▼",
            va="bottom", ha="right", fontsize="small", color = "red",
            bbox={'facecolor' : 'white', 'edgecolor' : 'none',
                  'pad': 0.5, 'alpha': 0.0})
    ax.set_ylim(min_value, max_value)
    ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Time")
    ax.set_ylabel("Error")
    ax.text(sample_size+500, 1e-3, "C")
    
    
    plt.tight_layout()
    f.savefig(args.o[0])
    #plt.show()

if __name__ == "__main__":
    main(sys.argv)

