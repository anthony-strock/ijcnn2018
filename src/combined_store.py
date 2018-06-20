#!/usr/bin/env python
import sys

import numpy as np
import matplotlib.pyplot as plt
from utils import plot

def main(seed = 0):

    ### Parameters of the plot

    np.random.seed(seed)
    n = 50
    s = 50
    figsize = (6, 2)
    p = 5
    
    ### Creating the data

    X = np.arange(n)
    I = np.random.uniform(-1, 1, n)
    
    T1 = np.zeros(n)
    T1[n//2] = 1
    D1 = np.zeros(n)
    D1[n//2:] = I[n//2]
    
    T2 = np.zeros(n)
    for t in range(n//2, n, p):
        T2[t] = 1
    D2 = np.zeros(n)
    for t in range(n//2, n, p):
        D2[t:] = I[t]
    
    T3 = np.zeros(n)
    T3[n//2:] = 1
    D3 = np.zeros(n)
    D3[n//2:] = I[n//2:]
    
    ### Plotting the data

    fig = plt.figure(figsize=figsize)
    
    ax = plt.subplot(3,1,1, frameon=False)
    ax.yaxis.set_ticks_position('none')
    plt.tick_params(axis='both', which='major', labelsize=7)
    plot(ax, I, y=3, s=s)
    plot(ax, T1, y=2, color="black", s=s)
    plot(ax, D1, y=1, s=s)
    plt.xlim(-0.5,n-0.5)
    plt.xticks([])
    plt.ylim(0.5, 3.5)
    plt.yticks([1, 2, 3],
               ["Desired", "Trigger", "Value"])
    plt.text(n,1.5, "A")
    
    
    ax = plt.subplot(3,1,2, frameon=False)
    ax.yaxis.set_ticks_position('none')
    plt.tick_params(axis='both', which='major', labelsize=7)
    plot(ax, I, y=3, s=s)
    plot(ax, T2, y=2, color="black", s=s)
    plot(ax, D2, y=1, s=s)
    plt.xlim(-0.5,n-0.5)
    plt.xticks([])
    plt.ylim(0.5, 3.5)
    plt.yticks([1, 2, 3],
               ["Desired", "Trigger", "Value"])
    plt.text(n,1.5,"B")
               
    ax = plt.subplot(3,1,3, frameon=False)
    ax.yaxis.set_ticks_position('none')
    plt.tick_params(axis='both', which='major', labelsize=7)
    plot(ax, I, y=3, s=s)
    plot(ax, T3, y=2, color="black", s=s)
    plot(ax, D3, y=1, s=s)
    plt.xlim(-0.5,n-0.5)
    plt.xticks([])
    plt.ylim(0.5, 3.5)
    plt.yticks([1, 2, 3],
               ["Desired", "Trigger", "Value"])
    plt.text(n,1.5, "C")

    plt.tight_layout()
    plt.savefig("combined_store_"+str(n)+"_"+str(seed)+".pdf", transparent = True, dpi = 600)    
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(seed = int(sys.argv[1]))
    else:
        main()
