import numpy as np

    

def whiten(x):
    if len(x.shape)==1:        
        x_m = np.mean(x)
        x_std = np.std(x)
        x_c = (x - x_m)
        x_cr = x_c/x_std
        return x_cr
    if len(x.shape)==2:
        x_m = np.mean(x, axis = 0)
        x_std = np.std(x, axis = 0)
        x_c = (x - x_m)
        x_cr = x_c/x_std
        return x_cr
    elif len(x.shape)==3:
        x_all = np.concatenate(x, axis = 0)
        x_all_m = np.mean(x_all, axis = 0)
        x_all_std = np.std(x_all, axis = 0)
        x_all_c = (x_all - x_all_m)
        x_all_cr = x_all_c/x_all_std
        return x_all_cr.reshape(x.shape)
    else:
        raise NameError("Not implemented yet")
        
def best_correlated(x,y,k=None):
    """    
    # Arguments
        x: the different evolutions over different dimensions, np.array, len(x.shape)=2,
           x.shape[0] represents the index of the time
           x.shape[1] represents the index of the dimension
        y: the different evolutions over one dimension, np.array, len(y.shape)=1,
           y.shape[0] represents the index of the time
        k: number of best correlated, None represents the maximal number
          
    # Returns
        The k 
    """
    assert(len(x.shape)==2)
    assert(len(y.shape)==1)
    assert(y.shape[0]==x.shape[0])
    corr_x_y = np.dot(whiten(x).T, whiten(y))/x.shape[0]
    abs_corr_x_y = np.abs(corr_x_y)
    idx = np.argsort(abs_corr_x_y)
    np
    if k != None:
        s_idx = idx[::-1][:k]
    else:
        s_idx = idx[::-1]
    return s_idx, corr_x_y[s_idx]
    
def reconstruction_error(x,y,warmup=0):
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
        The reconstruction error from x to y
    """
    assert(len(x.shape)==3)
    assert(len(y.shape)==3)
    assert(x.shape[:2]==y.shape[:2])
    X = np.empty((x.shape[0]*(x.shape[1]-warmup),x.shape[2]+1))
    X[:,0] = 1
    X[:,1:] = np.concatenate(x[:,warmup:], axis = 0)
    Y = np.concatenate(y[:,warmup:], axis = 0)
    A = np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)),X.T),Y)
    r_Y = np.dot(X,A)
    abs_Error = np.abs(r_Y-Y)
    abs_error = abs_Error.reshape(y.shape[0], y.shape[1]-warmup, y.shape[2])
    rms = np.sqrt(np.sum(abs_error**2, axis = (1,2))/(abs_error.shape[1]))
    return np.mean(rms), np.std(rms), A[0], A[1:]
    
def spectral_radius(w):
    return np.max(np.abs(np.linalg.eig(w)[0]))
