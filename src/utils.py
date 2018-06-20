import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import to_rgba as rgba
import time

def set_seed(seed = None, old_rstate = None, verbose = 0):
    if seed == None:
        seed = int(time.time())
    if verbose > 0:
        print("Seed:", seed)
    if old_rstate == None:
        return seed, np.random.RandomState(seed)
    else:
        old_rstate.seed(seed)
        return seed, old_rstate

def plot(ax, V, y=0, color=None, clip=None, s = 50):
    if clip is not None:
        x0, x1 = -1, len(V)+1
        y0, y1 = y, y+clip
        path = Path([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
        patch = PathPatch(path, facecolor='none', edgecolor='none')
        ax.add_patch(patch)
    else:
        patch = None

    I_nan     = np.argwhere(np.isnan(V))
    I_pos     = np.argwhere(V >= 0)
    I_pos_bad = np.argwhere(V > +1)
    I_neg     = np.argwhere(V <  0)
    I_neg_bad = np.argwhere(V < -1)
    
    if color is None:
        blue = rgba("#1f77b4")
        red  = rgba("#d62728")
    else:
        blue = rgba(color)
        red  = rgba(color)
    gray = rgba("0.85")

    # Positive values
    n = len(I_pos)
    if n > 0:
        X = np.squeeze(np.array(I_pos))
        Y = y*np.ones(n)
        C = np.zeros((n,4))
        C[:] = blue
        C[:,3] = np.squeeze(V[I_pos])
        ax.scatter(X, Y, s=s, facecolor=C, edgecolor=(0,0,0,.75), linewidth=0.5,
                   clip_path = patch)

    # Negative values
    n = len(I_neg)
    if n > 0:
        X = np.squeeze(np.array(I_neg))
        Y = y*np.ones(n)
        C = np.zeros((n,4))
        C[:] = red
        C[:,3] = np.squeeze(-V[I_neg])
        ax.scatter(X, Y, s=s, facecolor=C, edgecolor=(0,0,0,.75), linewidth=0.5,
                   clip_path = patch)

    # NaN values
    n = len(I_nan)
    if n > 0:
        X = I_nan
        Y = y*np.ones(n)
        ax.scatter(X, Y, s=s, facecolor=gray, edgecolor=gray, linewidth=0.5,
                   clip_path = patch)

if __name__ == "__main__":
    print("That's not executable")
