import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LogNorm

from .maths import whiten, reconstruction_error

def plot_2d_all_runs(x, y, xlabel = "", ylabel = "", xlim = None, ylim = None, figsize = None):
    """    
    # Arguments
        x: the different evolutions over x-axes, np.array, len(x.shape)=2,
           x.shape[0] represents the index of the sequence
           x.shape[1] represents the index of the time
        y: the different evolutions over y-axes, np.array, y.shape=x.shape
           y.shape[0] represents the index of the sequence
           y.shape[1] represents the index of the time
        
    # Returns
        The figure of the plot drawing the different trajectories
    """
    assert(len(x.shape)==2)
    assert(x.shape==y.shape)
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax = f.add_subplot(1,1,1)
    for i in range(x.shape[0]):
        points = np.array([x[i], y[i]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='Spectral')
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim!=None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(x.min(), x.max())
    if ylim!=None:
        ax.set_ylim(*ylim)
    else:        
        ax.set_ylim(y.min(), y.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return f
    
def plot_2d_all_runs_color(x, y, c, xlabel = "", ylabel = "", xlim = None, ylim = None, clim = None, figsize = None, clabel = ""):
    """    
    # Arguments
        x: the different evolutions over x-axes, np.array, len(x.shape)=2,
           x.shape[0] represents the index of the sequence
           x.shape[1] represents the index of the time
        y: the different evolutions over y-axes, np.array, y.shape=x.shape
           y.shape[0] represents the index of the sequence
           y.shape[1] represents the index of the time
        c: the different evolutions of the color, np.array, color.shape=x.shape
           c.shape[0] represents the index of the sequence
           c.shape[1] represents the index of the time
        
    # Returns
        The figure of the plot drawing the different trajectories colored given
        the color
    """
    if type(x)!= list and type(y)!= list and type(c)!= list:
        assert(len(x.shape)==2)
        assert(x.shape==y.shape)
        assert(x.shape==c.shape)
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax = f.add_subplot(1,1,1)
    if clim!=None:
        norm = plt.Normalize(*clim)
    else:
        norm = plt.Normalize(c.min(), c.max())
    for i in range(len(x)):
        points = np.array([x[i], y[i]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='Spectral', norm=norm)
        lc.set_array(c[i])
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
    cbar = f.colorbar(line, ax=ax)
    cbar.set_label(clabel, rotation=270)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim!=None:
        print("blou")
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(x.min(), x.max())
    if ylim!=None:
        ax.set_ylim(*ylim)
    else:        
        ax.set_ylim(y.min(), y.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return f
    
def imshow_1d_one_run(x, xlim = None, figsize = None, cmap = "seismic"):
    """    
    # Arguments
        x: the different evolutions over different dimensions, np.array, len(x.shape)=2,
           x.shape[0] represents the index of the time
           x.shape[1] represents the index of the dimension
        
    # Returns
        The figure of the image drawing the variation of x over time, each
        dimension being represented by a line of the image
    """
    assert(len(x.shape)==2)
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax= f.add_subplot(1,1,1)
    if xlim!=None:
        norm = plt.Normalize(*xlim)
    else:
        norm = plt.Normalize(x.min(), x.max())
    obj = ax.imshow(x.T, interpolation='nearest', cmap=cmap, norm = norm, aspect = "auto")
    #ax.set_ylim([0,x.shape[1]-1])
    cb = f.colorbar(obj)
    return f

def plot_1d_one_run(x, xlim = None, figsize = None, scale = "linear", tlim = None, color = "black", alpha = 1.0, xlabel = "", tlabel = "Time"):
    """    
    # Arguments
        x: the different evolutions over different dimensions, np.array, len(x.shape)=2,
           x.shape[0] represents the index of the time
           x.shape[1] represents the index of the dimension
        
    # Returns
        The figure of the plot of the variation of x over time
    """
    assert(len(x.shape)==2)
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax= f.add_subplot(1,1,1)
    ax.set_yscale(scale)
    if xlim != None:
        ax.set_ylim(*xlim)
    if tlim != None:
        ax.set_xlim(*tlim)
    ax.set_xlabel(tlabel)
    ax.set_ylabel(xlabel)
    for k in range(x.shape[1]):
        ax.plot(x[:,k], color = color, alpha = alpha)
    return f



def plot_1d_all_runs(x, xlim = None, figsize = None, scale = "linear", color = "black", alpha = 0.1, tlim = None, linestyle = "-", xlabel = "", tlabel = "Time"):
    """    
    # Arguments
        x: the different evolutions over different dimensions, np.array, len(x.shape)=3,
           x.shape[0] represents the index of the sequence
           x.shape[1] represents the index of the time
           x.shape[2] represents the index of the dimension
        
    # Returns
        The figure of the plot of the variation of x over time
    """
    assert(len(x.shape)==3)
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax= f.add_subplot(1,1,1)
    ax.set_yscale(scale)
    if xlim != None:
        ax.set_ylim(*xlim)
    if tlim != None:
        ax.set_xlim(*tlim)
    ax.set_xlabel(tlabel)
    ax.set_ylabel(xlabel)
    for s in range(x.shape[0]):
        for k in range(x.shape[2]):
            ax.plot(x[s,:,k], color = color, alpha = alpha)
    return f
    
def scatter_1d_all_runs(x, xlim = None, figsize = None, scale = "linear", color = "black", alpha = 0.1, tlim = None, linestyle = "-"):
    """    
    # Arguments
        x: the different evolutions over different dimensions, np.array, len(x.shape)=3,
           x.shape[0] represents the index of the sequence
           x.shape[1] represents the index of the time
           x.shape[2] represents the index of the dimension
        
    # Returns
        The figure of the plot of the variation of x over time
    """
    assert(len(x.shape)==3)
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax= f.add_subplot(1,1,1)
    ax.set_yscale(scale)
    if xlim != None:
        ax.set_ylim(*xlim)
    if tlim != None:
        ax.set_xlim(*tlim)
    time = np.arange(x.shape[1])
    for s in range(x.shape[0]):
        for k in range(x.shape[2]):
            ax.scatter(time, x[s,:,k], color = color, alpha = alpha)
    return f

def boxplot_1d_all_runs(x, xlim = None, figsize = None, scale = "linear", xlabel = ""):
    """    
    # Arguments
        x: the different evolutions over different dimensions, np.array, len(x.shape)=2,
           x.shape[0] represents the index of the sequence
           x.shape[1] represents the index of the time
        
    # Returns
        The figure of the plot of the variation of x over time
    """
    print(x.shape)
    assert(len(x.shape)==2)
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax= f.add_subplot(1,1,1)
    ax.set_yscale(scale)
    ax.set_xlabel("Time")
    if xlabel != None:
        ax.set_ylabel(xlabel)
    percentiles = [5,50,95]
    percentiles_x = np.percentile(x, percentiles, axis = 0)
    time = np.arange(x.shape[1])
    ax.fill_between(time, percentiles_x[0], percentiles_x[2], facecolor = "black", edgecolor = None, alpha = 0.1)
    ax.plot(time, percentiles_x[1], linestyle = "-", linewidth=0.5, color = "black")
    return f

def hist_cross_correlation(x, y, n_bins, figsize = None):
    """    
    # Arguments
        x: the different evolutions over different dimensions, np.array, len(x.shape)=2,
           x.shape[0] represents the index of the time
           x.shape[1] represents the index of the dimension
        y: the different evolutions over one dimension, np.array, len(y.shape)=1,
           y.shape[0] represents the index of the time
        n_bins: The number of bins used for the histogram
        
    # Returns
        The figure of the hist of the correlation between the different dimension of x and y
    """
    corr_x_y = np.dot(whiten(x).T, whiten(y))/x.shape[0]
    corr_x_y[np.isnan(corr_x_y)] = 0
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax= f.add_subplot(1,1,1)
    ax.set_ylabel("Number")
    ax.set_xlabel("Correlation")
    print(corr_x_y.shape)
    obj = ax.hist(corr_x_y,  -1+2*np.arange(n_bins)/(n_bins-1))
    ax.set_xlim(-1,1)
    return f

def plot_autocorrelation(x, dmax, figsize = None, cross = True):
    """    
    # Arguments
        x: the different evolutions over different dimensions, np.array, len(x.shape)=3,
           x.shape[0] represents the index of the sequence
           x.shape[1] represents the index of the time
           x.shape[2] represents the index of the dimension
        
    # Returns
        The figure of the hist of the autocorrelation of x, with different delays
    """
    assert(len(x.shape)==3)
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax= f.add_subplot(1,1,1)
    whiten_x = whiten(np.concatenate(x, axis = 0))
    autocorr_x = np.dot(whiten_x.T, whiten_x)/(x.shape[0]*x.shape[1])
    norm = plt.Normalize(-1, 1)
    if cross:
        obj = ax.imshow(autocorr_x , interpolation='nearest', cmap='Spectral', norm = norm)
    else:    
        obj = ax.imshow(np.diag(autocorr_x).reshape(1,x.shape[2]) , interpolation='nearest', cmap='Spectral', norm = norm)
    cb = f.colorbar(obj)
    cb.set_label("delay 0")
    
    def update(i, x, obj, cb, cross):
        if i == 0:
            return obj,
        cb.set_label("delay {:d}".format(i))
        whiten_x_d0 = whiten(np.concatenate(x[:,:-i], axis = 0))
        whiten_x_di = whiten(np.concatenate(x[:,i:], axis = 0))
        autocorr_x = np.dot(whiten_x_d0.T, whiten_x_di)/((x.shape[0]-i)*x.shape[1])
        if cross:
            obj.set_array(autocorr_x)
        else:
            obj.set_array(np.diag(autocorr_x).reshape(1,x.shape[2]))
        return obj, 
    ani = FuncAnimation(f, update, frames=dmax, fargs = (x,obj,cb,cross), interval=500, blit=True)
    return f,ani
    
def plot_cross_correlation(x,y,dmax, figsize = None):
    """    
    # Arguments
        x: the different evolutions over different dimensions, np.array, len(x.shape)=3,
           x.shape[0] represents the index of the sequence
           x.shape[1] represents the index of the time
           x.shape[2] represents the index of the dimension
        y: the different evolutions over different dimensions, np.array, len(y.shape)=3, y.shape[:2]=x.shape[:2],
           y.shape[0] represents the index of the sequence
           y.shape[1] represents the index of the time
           y.shape[2] represents the index of the dimension
          
    # Returns
        The figure of the hist of the crosscorrelation between x and y, with different delays
    """
    assert(len(x.shape)==3)
    assert(len(y.shape)==3)
    assert(y.shape[:2]==x.shape[:2])
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax= f.add_subplot(1,1,1)
    whiten_x = whiten(np.concatenate(x, axis = 0))
    whiten_y = whiten(np.concatenate(y, axis = 0))
    corr_x_y = np.dot(whiten_x.T, whiten_y)/(x.shape[0]*x.shape[1])
    norm = plt.Normalize(-1, 1)
    obj = ax.imshow(corr_x_y.T , interpolation='nearest', cmap='Spectral', norm = norm)
    cb = f.colorbar(obj)
    cb.set_label("delay 0")
    
    def update(i, x, obj, cb):
        if i == 0:
            return obj,
        cb.set_label("delay {:d}".format(i))
        whiten_x_d0 = whiten(np.concatenate(x[:,:-i], axis = 0))
        whiten_y_di = whiten(np.concatenate(y[:,i:], axis = 0))
        corr_x_y = np.dot(whiten_x_d0.T, whiten_y_di)/((x.shape[0]-i)*x.shape[1])
        obj.set_array(corr_x_y.T)
        return obj,
    ani = FuncAnimation(f, update, frames=dmax, fargs = (x,obj,cb), interval=500, blit=True)
    return f,ani
    
def plot_reconstruction_error(x,y,warmup=0,figsize=None, ax = None):
    """    
    # Arguments
        x: signal used to reconstruct, np.array, len(x.shape)=3,
           x.shape[0] represents the index of the sequence
           x.shape[1] represents the index of the time
           x.shape[2] represents the index of the dimension
        y: signal to reconstruct, np.array, len(y.shape)=3, y.shape[:2]=x.shape[:2],
           y.shape[0] represents the index of the sequence
           y.shape[1] represents the index of the time
           y.shape[2] represents the index of the dimension
          
    # Returns
        The figure of the hist of the crosscorrelation between x and y, with different delays
    """
    assert(len(x.shape)==3)
    assert(len(y.shape)==3)
    assert(y.shape[:2]==x.shape[:2])
    mean_error = np.empty(x.shape[2])
    std_error = np.empty(x.shape[2])
    for k in range(x.shape[2]):
        mean_error[k], std_error[k], _, _ = reconstruction_error(x[:,:,:k],y,warmup=warmup)
    if ax !=None:
        f = ax.get_figure()
    else:
        if figsize!=None:
            f = plt.figure(figsize=figsize)
        else:
            f = plt.figure()
        ax = f.add_subplot(1,1,1)
    ax.plot(np.arange(1,x.shape[2]+1),mean_error)
    ax.set_xlabel("number of dimensions used")
    ax.set_yscale("log")
    #=ax.set_ylabel("mean reconstruction rmse")
    return f
    
def scatter_2d_all_runs(x, y, xlabel = "", ylabel = "", xlim = None, ylim = None, figsize = None, s = 15.0):
    """    
    # Arguments
        x: the different evolutions over x-axes, np.array, len(x.shape)=2,
           x.shape[0] represents the index of the sequence
           x.shape[1] represents the index of the time
        y: the different evolutions over y-axes, np.array, y.shape=x.shape
           y.shape[0] represents the index of the sequence
           y.shape[1] represents the index of the time
        
    # Returns
        The figure of the plot drawing the different trajectories
    """
    assert(len(x.shape)==2)
    assert(x.shape==y.shape)
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax = f.add_subplot(1,1,1)
    for i in range(x.shape[0]):
        ax.scatter(x[i],y[i], color = "blue", s = s, alpha = 0.5, edgecolor = "none")
    if xlim!=None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(x.min(), x.max())
    if ylim!=None:
        ax.set_ylim(*ylim)
    else:        
        ax.set_ylim(y.min(), y.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return f

def hist_temporal_variation(x, n_bins, figsize = None, scale = "linear", xlim = None):
    """    
    # Arguments
        x: signal used to reconstruct, np.array, len(x.shape)=2 or len(x.shape)=3,
           x.shape[0] represents the index of the sequence
           x.shape[1] represents the index of the time
           if x.shape[2] exists it represents the index of the dimension
        n_bins: The number of bins used for the histogram   
            
    # Returns
        The figure of the plot drawing the histogram of the maximal temporal variations in each of the sequences
    """
    assert(len(x.shape)==3 or len(x.shape)==2)
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax = f.add_subplot(1,1,1)
    max_x = np.max(x, axis = 1)
    min_x = np.min(x, axis = 1)
    variation = (max_x-min_x).flatten()
    
    ax.set_xscale(scale)
    if xlim != None:
        ax.set_xlim(*xlim)
        ax.hist(variation, xlim[0]+(xlim[1]-xlim[0])*np.arange(n_bins)/(n_bins-1))
    else:
        ax.hist(variation, n_bins)
    return f

def hist_values(x, n_bins, figsize = None, scale = "linear", xlim = None, xlabel = ""):
    """    
    # Arguments
        x: signal used to reconstruct, np.array, len(x.shape)=2 or len(x.shape)=3,
           x.shape[0] represents the index of the sequence
           if x.shape[1] exists it represents the index of the time
           if x.shape[2] exists it represents the index of the dimension
        n_bins: The number of bins used for the histogram   
            
    # Returns
        The figure of the plot drawing the histogram of the values taken by all the sequence in x at all time in all dimension
    """
    assert(len(x.shape)==3 or len(x.shape)==2  or len(x.shape)==2)
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax = f.add_subplot(1,1,1)    
    ax.set_xscale(scale)
    ax.set_ylabel("Number")
    ax.set_xlabel(xlabel)
    if xlim != None:
        ax.set_xlim(*xlim)
        ax.hist(x.flatten(), xlim[0]+(xlim[1]-xlim[0])*np.arange(n_bins)/(n_bins-1))
    else:
        ax.hist(x.flatten(), n_bins)
    return f

def scatter_heatmap(x, y, z, xlim=None, ylim=None, zlim=None, figsize = None, xlabel = "", ylabel = "", zlabel = "", cmap = "inferno", alpha = 1.0, xscale = "linear", yscale = "linear", zscale = "linear"):
    """    
    # Arguments
        x: np.array, len(x.shape)=1,
        y: np.array, y.shape == x.shape
        
    # Returns
        The figure of the heatmap of z in the coordinate of x and y
    """
    assert(len(x.shape)==1)
    assert(x.shape==y.shape)
    assert(x.shape==z.shape)
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax = f.add_subplot(1,1,1)
    if xlim!=None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(x.min(), x.max())
    if ylim!=None:
        ax.set_ylim(*ylim)
    else:        
        ax.set_ylim(y.min(), y.max())
    if zlim==None:
        zlim = [z.min(), z.max()]
    if zscale == "log":
        norm = LogNorm(*zlim)
    else:
        norm = plt.Normalize(*zlim)
    zmin, zmax = z.min(),z.max()
    s = (z-zmin)/(zmax-zmin)
    cmap = plt.get_cmap(cmap, 10)
    obj = ax.scatter(x, y, s = 20+s*100, c = z, alpha = alpha, edgecolor = None, cmap = cmap, norm = norm)
    f.colorbar(obj, ax = ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    return f

def matshow(w, figsize = None,  cmap = "inferno", wlim = None):
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()    
    if wlim!=None:
        norm = plt.Normalize(*wlim)
    else:
        norm = plt.Normalize(w.min(), w.max())
    ax = f.add_subplot(1,1,1)
    obj = ax.imshow(w, cmap = cmap, norm = norm)
    cb = f.colorbar(obj)
    return f
    
def plot_3d_all_runs_color(x, y, z, c, xlabel = "", ylabel = "", zlabel = "", xlim = None, ylim = None, zlim = None, clim = None, figsize = None):
    """    
    # Arguments
        x: the different evolutions over x-axes, np.array, len(x.shape)=2,
           x.shape[0] represents the index of the sequence
           x.shape[1] represents the index of the time
        y: the different evolutions over y-axes, np.array, y.shape=x.shape
           y.shape[0] represents the index of the sequence
           y.shape[1] represents the index of the time
        z: the different evolutions over y-axes, np.array, z.shape=x.shape
           z.shape[0] represents the index of the sequence
           z.shape[1] represents the index of the time
        c: the different evolutions of the color, np.array, color.shape=x.shape
           c.shape[0] represents the index of the sequence
           c.shape[1] represents the index of the time
        
    # Returns
        The figure of the plot drawing the different trajectories colored given
        the color
    """
    assert(len(x.shape)==2)
    assert(x.shape==y.shape)
    assert(x.shape==z.shape)
    assert(x.shape==c.shape)
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax = f.gca(projection='3d')
    if clim!=None:
        norm = plt.Normalize(*clim)
    else:
        norm = plt.Normalize(c.min(), c.max())
    for i in range(x.shape[0]):
        points = np.array([x[i], y[i], z[i]]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, cmap='Spectral', norm=norm)
        lc.set_array(c[i])
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
    f.colorbar(line, ax=ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim!=None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(x.min(), x.max())
    if ylim!=None:
        ax.set_ylim(*ylim)
    else:        
        ax.set_ylim(y.min(), y.max())
    if zlim!=None:
        ax.set_zlim(*zlim)
    else:        
        ax.set_zlim(z.min(), z.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    return f
    
def scatter_3d_all_runs_color(x, y, z, c, xlabel = "", ylabel = "", zlabel = "", clabel = "", xlim = None, ylim = None, zlim = None, clim = None, figsize = None, s = 15.0):
    """    
    # Arguments
        x: the different evolutions over x-axes, np.array, len(x.shape)=2,
           x.shape[0] represents the index of the sequence
           x.shape[1] represents the index of the time
        y: the different evolutions over y-axes, np.array, y.shape=x.shape
           y.shape[0] represents the index of the sequence
           y.shape[1] represents the index of the time
        z: the different evolutions over y-axes, np.array, z.shape=x.shape
           z.shape[0] represents the index of the sequence
           z.shape[1] represents the index of the time
        c: the different evolutions of the color, np.array, color.shape=x.shape
           c.shape[0] represents the index of the sequence
           c.shape[1] represents the index of the time
        
    # Returns
        The figure of the plot drawing the different trajectories colored given
        the color
    """
    assert(len(x.shape)==2)
    assert(x.shape==y.shape)
    assert(x.shape==z.shape)
    assert(x.shape==c.shape)
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax = f.gca(projection='3d')
    if clim!=None:
        norm = plt.Normalize(*clim)
    else:
        norm = plt.Normalize(c.min(), c.max())
        
    for i in range(x.shape[0]):
        obj = ax.scatter(x[i],y[i], z[i], c = c[i], cmap ='Spectral', norm = norm, s = s, alpha = 0.5, edgecolors = None, linewidth = 0)
    cbar = f.colorbar(obj, ax=ax)
    cbar.set_label(clabel, rotation=90)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim!=None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(x.min(), x.max())
    if ylim!=None:
        ax.set_ylim(*ylim)
    else:        
        ax.set_ylim(y.min(), y.max())
    if zlim!=None:
        ax.set_zlim(*zlim)
    else:        
        ax.set_zlim(z.min(), z.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    return f
    
def scatter_3d_all_runs(x, y, z, xlabel = "", ylabel = "", zlabel = "", xlim = None, ylim = None, zlim = None, figsize = None, s = 15.0, c = "blue"):
    """    
    # Arguments
        x: the different evolutions over x-axes, np.array, len(x.shape)=2,
           x.shape[0] represents the index of the sequence
           x.shape[1] represents the index of the time
        y: the different evolutions over y-axes, np.array, y.shape=x.shape
           y.shape[0] represents the index of the sequence
           y.shape[1] represents the index of the time
        z: the different evolutions over y-axes, np.array, z.shape=x.shape
           z.shape[0] represents the index of the sequence
           z.shape[1] represents the index of the time
        c: the different evolutions of the color, np.array, color.shape=x.shape
           c.shape[0] represents the index of the sequence
           c.shape[1] represents the index of the time
        
    # Returns
        The figure of the plot drawing the different trajectories colored given
        the color
    """
    assert(len(x.shape)==2)
    assert(x.shape==y.shape)
    assert(x.shape==z.shape)
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax = f.gca(projection='3d')
        
    for i in range(x.shape[0]):
        obj = ax.scatter(x[i],y[i], z[i], c = "blue", s = s, alpha = 0.5, edgecolors = None, linewidth = 0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim!=None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(x.min(), x.max())
    if ylim!=None:
        ax.set_ylim(*ylim)
    else:        
        ax.set_ylim(y.min(), y.max())
    if zlim!=None:
        ax.set_zlim(*zlim)
    else:        
        ax.set_zlim(z.min(), z.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    return f

def scatter_2d_all_runs_color(x, y, c, xlabel = "", ylabel = "", xlim = None, ylim = None, clim = None, figsize = None, s = 15, alpha = 0.5, clabel = "", crotation = 270, showbar = True):
    """    
    # Arguments
        x: the different evolutions over x-axes, np.array, len(x.shape)=2,
           x.shape[0] represents the index of the sequence
           x.shape[1] represents the index of the time
        y: the different evolutions over y-axes, np.array, y.shape=x.shape
           y.shape[0] represents the index of the sequence
           y.shape[1] represents the index of the time
        c: the different evolutions of the color, np.array, color.shape=x.shape
           c.shape[0] represents the index of the sequence
           c.shape[1] represents the index of the time
        
    # Returns
        The figure of the plot drawing the different trajectories colored given
        the color
    """
    assert(len(x.shape)==2)
    assert(x.shape==y.shape)
    assert(x.shape==c.shape)
    if figsize!=None:
        f = plt.figure(figsize=figsize)
    else:
        f = plt.figure()
    ax = f.add_subplot(1,1,1)
    if clim!=None:
        norm = plt.Normalize(*clim)
    else:
        norm = plt.Normalize(c.min(), c.max())
    for i in range(x.shape[0]):
        obj = ax.scatter(x[i],y[i], c = c[i], cmap ='Spectral', norm = norm, s = s, alpha = alpha, edgecolors = "none", linewidths = 0)
    if showbar:
        cbar = f.colorbar(obj, ax=ax)
        cbar.set_label(clabel, rotation=crotation)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim!=None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(x.min(), x.max())
    if ylim!=None:
        ax.set_ylim(*ylim)
    else:        
        ax.set_ylim(y.min(), y.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return f
