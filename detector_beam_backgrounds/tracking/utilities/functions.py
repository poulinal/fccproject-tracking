#from podio import root_io
#import ROOT
#import math
#import numpy as np
#import os


import matplotlib.pyplot as plt
import mplhep as hep 
#import seaborn as sns
#import pickle
#import hist

hep.style.use(hep.style.ROOT)

'''
def phi(x,y):
    """
    Calculates phi of particle.
    Inputs: x,y floats.
    Output: phi, float representing angle in radians from 0 to 2 pi.
    """
    phi = math.atan(y/x)
    if x < 0:
        phi +=  math.pi
    elif y < 0:
        phi += 2*math.pi
    return phi

def theta(x,y,z):
    """
    Calculates theta of particle.
    Inputs: x,y,z floats.
    Output: theta, float representing angle in radians from 0 to pi.
    """
    return math.acos(z/np.sqrt(x**2 + y**2 + z**2))


def radius_idx(hit, layer_radii):
    """
    Calculates polar radius of particle.
    Inputs: hit, SimTrackerHit object.
    Output: r, int representing polar radius in mm.
    """
    true_radius = hit.rho()
    for i,r in enumerate(layer_radii):
        if abs(true_radius-r) < 4:
            return i
    raise ValueError(f"Not close enough to any of the layers {true_radius}")
'''

def plot_hist(h, outname, title, xMin=-1, xMax=-1, yMin=-1, yMax=-1, xLabel="", yLabel="Events", logY=False):
    fig = plt.figure()
    ax = fig.subplots()

    hep.histplot(h, label="", ax=ax, yerr=False)

    ax.set_title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    #ax.legend(fontsize='x-small')
    if logY:
        ax.set_yscale("log")

    if xMin != -1 and xMax != -1:
        ax.set_xlim([xMin, xMax])
    if yMin != -1 and yMax != -1:
        ax.set_ylim([yMin, yMax])


    fig.savefig(outname, bbox_inches="tight")
    fig.savefig(outname.replace(".png", ".pdf"), bbox_inches="tight")



def plot_2dhist(h, outname, title, xMin=-1, xMax=-1, yMin=-1, yMax=-1, xLabel="", yLabel="Events", logY=False):
    fig = plt.figure()
    ax = fig.subplots()

    hep.hist2dplot(h, label="", ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    #ax.legend(fontsize='x-small')
    if logY:
        ax.set_yscale("log")

    if xMin != -1 and xMax != -1:
        ax.set_xlim([xMin, xMax])
    if yMin != -1 and yMax != -1:
        ax.set_ylim([yMin, yMax])


    fig.savefig(outname, bbox_inches="tight")
    fig.savefig(outname.replace(".png", ".pdf"), bbox_inches="tight")

    
def hist_plot(h, outname, title, xMin=-1, xMax=-1, yMin=-1, yMax=-1, xLabel="", yLabel="Events", logY=False):
    #recreate plot_hist but using hist instead of hep
    fig = plt.figure()
    
    ax = fig.subplots()
    
    ax.hist(h, bins=100, histtype='step', label='MC particles')
    
    ax.set_title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.legend(fontsize='x-small')
    if logY:
        ax.set_yscale("log")
    
    if xMin != -1 and xMax != -1:
        ax.set_xlim([xMin, xMax])
    if yMin != -1 and yMax != -1:
        ax.set_ylim([yMin, yMax])
    
    fig.savefig(outname, bbox_inches="tight")
    fig.savefig(outname.replace(".png", ".pdf"), bbox_inches="tight")
    
'''
#make same as plot_hist
def sns_plot(h, outname, title, xMin=-1, xMax=-1, yMin=-1, yMax=-1, xLabel="", yLabel="Events", logY=False):
    fig = plt.figure()
    
    ax = fig.subplots()


    sns.histplot(h, bins=100, histtype='step', label='MC particles')
    
    ax.set_title(title)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.legend(fontsize='x-small')
    if logY:
        ax.set_yscale("log")
    
    if xMin != -1 and xMax != -1:
        ax.set_xlim([xMin, xMax])
    if yMin != -1 and yMax != -1:
        ax.set_ylim([yMin, yMax])
    
    fig.savefig(outname, bbox_inches="tight")
    fig.savefig(outname.replace(".png", ".pdf"), bbox_inches="tight")
'''