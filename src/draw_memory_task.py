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
    parser.add_argument("inputs", nargs = 1, type = str, help = "path to the file where the input is saved")
    parser.add_argument("outputs", nargs = 1, type = str, help = "path to the file where the output is saved")
    parser.add_argument("desired", nargs = 1, type = str, help = "path to the file where the desired output is saved")
    parser.add_argument("-o", nargs = 1, type = str, help = "path to the file where the figure will be saved")
    parser.add_argument("-r", nargs = "?", default = 0, type = int, help = "index of reservoir")
    parser.add_argument("-i", nargs = "?", default = 0, type = int, help = "index of test sequence")
    parser.add_argument("--log", action = "store_true", help = "logscale for error")
    args = parser.parse_args(argv[1:])
    # Parameters
    # --------------------------------------------------------------------------
    trigger_time = 100
    trigger_step = 1000
    
    # Loading data
    # --------------------------------------------------------------------------    
    inputs = np.load(args.inputs[0])
    outputs = np.load(args.outputs[0])
    desired_outputs = np.load(args.desired[0])

    sample_size = inputs.shape[1]

    time = np.arange(inputs.shape[1])
    
    warmup = 30
    print(inputs.shape)
    
    # Display Once
    # --------------------------------------------------------------------------
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    plt.rc("ytick", labelsize="x-small")
    plt.rc("xtick", labelsize="x-small")
    
    f = plt.figure(figsize=(5,5))
    
    ax = f.add_subplot(4,1,1)
    X = np.arange(sample_size)
    ax.plot(time, inputs[args.i,:,0], linewidth = 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Value")
    
    idx, = np.where(inputs[args.i,:,1]==1)
    print(idx.shape)
    triggers = np.empty(3*len(idx)+2)
    time_triggers = np.empty(3*len(idx)+2)
    time_triggers[0] = 0
    triggers[0] = 0
    time_triggers[-1] = 100
    triggers[-1] = 0
    for i in range(len(idx)):
        triggers[1+3*i:1+3*(i+1)] = [0,1,0]
        time_triggers[1+3*i:1+3*(i+1)] = idx[i]
    ax = f.add_subplot(4,1,2)
    X = np.arange(sample_size)
    ax.plot(time_triggers, triggers, linewidth = 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Trigger")
    
    ax = f.add_subplot(4,1,3)
    X = np.arange(sample_size)
    ax.plot(time[warmup:], desired_outputs[args.i,warmup:,0], linewidth = 1.0)
    ax.plot(time, outputs[args.i,:,0], color = "red", linewidth = 1.0, linestyle='dashed')
    ax.set_ylim(-0.4, 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Memory")

    ax = f.add_subplot(4,1,4)
    X = np.arange(sample_size)
    ax.plot(time[:warmup], np.abs(outputs[args.i,:warmup,0]-desired_outputs[args.i,:warmup,0]), color = "green" , linewidth = 1.0, linestyle="dashed")
    ax.plot(time[warmup:], np.abs(outputs[args.i,warmup:,0]-desired_outputs[args.i,warmup:,0]), color = "green" , linewidth = 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Error")
    if args.log:
        ax.set_ylim(1e-6, 1e-1)
        ax.set_yscale("log")
    else:
        ax.set_ylim(0.0, 2e-4)
        

    
    plt.tight_layout()
    f.savefig(args.o[0], transparent = True)
    #plt.show()

if __name__ == "__main__":
    main(sys.argv)

