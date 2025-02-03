#Alexander Poulin Jan 2025
import ROOT
import numpy as np 
from utilities.functions import hist_plot, plot_hist, multi_hist_plot, \
    bar_plot, multi_bar_plot, xy_plot, bar_step_multi_hist_plot
from utilities.pltWireCh import plot_wire_chamber
import argparse
import sys
import matplotlib.pyplot as plt
import math

"""
This file contains functions to plot the background data.
The functions are as follows:
hitsPerMC -- Given that there was a hit for this particle, count the number of times that particle hits the detector.
momPerMC -- Given a hit by a particle, what is that particles momentum.
PDGPerMC -- Given a hit, what is the PDG of the particle which produced that hit.
occupancy -- Determines occupancy and related values.
plotMomentum -- Plot the momentum of all particles.
plotPDG -- Plot the PDG of all particles.
plotHits -- Plot the number of hits per particle.
plotOccupancy -- Plot the occupancy of the detector.

See the function docstrings for more information on each function.
See bottom for example usage and documentation of argument parsing.
"""



available_functions = ["hitsPerMC", "momPerMC", "PDGPerMC", "wiresPerMC", "trajLen", "radiusPerMC", "angleHits", "occupancy"]
dic = {}
dicbkg = {}
numFiles = 500
# backgroundDataPath = "fccproject-tracking/detector_beam_backgrounds/tracking/data/bkg_background_particles_"+str(numFiles)+".npy"
backgroundDataPath = "fccproject-tracking/detector_beam_backgrounds/tracking/data/occupancy_tinker/bkg_background_particles_"+str(numFiles)+".npy"
combinedDataPath = "fccproject-tracking/detector_beam_backgrounds/tracking/data/combined/"
imageOutputPath = "fccproject-tracking/detector_beam_backgrounds/tracking/images/test"
signalDataPath = "fccproject-tracking/detector_beam_backgrounds/tracking/data/occupancy_tinker/signal_background_particles_"+str(numFiles)+".npy"
typeFile="Bkg"#dont change

#change to personal directories in here:
def setup(typeFile: str =typeFile, includeBkg: bool =False):
    """
    Setups the file paths and outputs.
    Data paths will lead to either the background or signal data or their combined files (not yet tested).
    Output paths should be entirely dependent on your directory and where you want to specify.
    
    Note data paths should point to a .npy file that contains a dictionary with the following keys: \n
        "hits" -- list of hits per mcParticle \n
        "pdg" -- list of pdg per mcParticle \n
        "p" -- list of momentum per mcParticle \n
        "px" -- list of px per mcParticle \n
        "py" -- list of py per mcParticle \n
        "pz" -- list of pz per mcParticle \n
        "gens" -- list of generator status per mcParticle \n
        "R" -- list of vertex radius per mcParticle \n
        "pos_z" -- list of z position of hits \n
        "hits_produced_secondary" -- list of hits produced by secondary particles \n
        "hits_mc_produced_secondary" -- list of mcParticles that produced secondary particles \n
        "has_par_photon" -- list of mcParticles that have a parent photon \n
        "count_hits" -- list of number of hits per mcParticle \n
        "pdg" -- list of pdg per mcParticle \n
        "list_n_cells_fired_mc" -- list of number of cells fired per mcParticle \n
        "percentage_of_fired_cells" -- list of percentage of cells fired per mcParticle \n
        "n_cell_per_layer" -- list of number of cells per layer \n
        "total_number_of_cells" -- total number of cells \n
        "total_number_of_layers" -- total number of layers \n
        "occupancy_per_batch_sum_events" -- list of occupancy per batch, summing occupancies an event \n
        "occupancy_per_batch_sum_events_error" -- list of occupancy per batch, summing occupancies an event \n
        "occupancy_per_batch_sum_events_non_normalized" -- list of occupancy per batch, summing occupancies an event \n
        "occupancy_per_batch_sum_events_non_normalized_error" -- list of occupancy per batch, summing occupancies an event \n
        "occupancy_per_batch_sum_batches" -- list of occupancy per batch, summing occupancies a batch \n
        "occupancy_per_batch_sum_batches_error" -- list of occupancy per batch, summing occupancies a batch \n
        "occupancy_per_batch_sum_batches_non_normalized" -- list of occupancy per batch, summing occupancies a batch \n
        "occupancy_per_batch_sum_batches_non_normalized_error" -- list of occupancy per batch, summing occupancies a batch \n
    
    Inputs:
        type -- what type of data to load, either Bkg, Combined, or Signal 
        includeBkg -- for when we want to overlay bkg and signal files in a plot 
    Return: no return, just updates the global dictionary dic
    """
    if typeFile == "Bkg":
        dic = np.load(backgroundDataPath, allow_pickle=True).item()
    elif typeFile == "Combined":
        dic = np.load(combinedDataPath, allow_pickle=True).item()
    elif typeFile == "Signal":
        dic = np.load(signalDataPath, allow_pickle=True).item()
    else:
        print("Type must be either background, combined, or signal")
        sys.exit()
    if includeBkg:
        dicbkg = np.load(backgroundDataPath, allow_pickle=True).item()

def hitsPerMC(dic, args = ""):
    """
    Given that there was a hit for this particle, count the number of times that particle hits the detector.
    Inputs: dic from setup,
            args is the type of hits to calculate:
                "" -- calculate all args \n
                "all" -- all particles \n
                "photon" -- all particles with pdg 22 \n
                "photonSec" -- all particles with pdg 22 and produced secondary particles \n
                "neutron" -- all particles with pdg 2112 \n
                "neutronSec" -- all particles with pdg 2112 and produced secondary particles \n
                "electron" -- all particles with pdg 11 \n
                "allPDG" -- all particles with pdg as key and count as value \n
                "multiHits" -- all particles with keys of one hit, >1 hit, >5 hits, >10 hits, >20 hits \n
    Outputs: hist, dictionary of values (count of hits) with keys: \n
        "All", "Only Photons","Only Photons Produced by Secondary Particles", 
        "Only Neutrons", "Only Neutrons Produced by Secondary Particles", 
        "Only Electrons", "Only Electrons Produced by Secondary Particles", 
        "[allPDG] -- various dic of pdg number", "1 Hit", ">1 Hits", ">5 Hits", ">10 Hits", ">20 Hits"
    """
    hist = {} #empty dictionary to hold all the calculations
    hist["All"] = []
    hist["Only Photons"] = []
    hist["Only Photons Produced by Secondary Particles"] = []
    hist["Only Neutrons"] = []
    hist["Only Neutrons Produced by Secondary Particles"] = []
    hist["Only Electrons"] = []
    hist["Only Electrons Produced by Secondary Particles"] = []
    hist["1 Hit"] = []
    hist[">1 Hits"] = []
    hist[">5 Hits"] = []
    hist[">10 Hits"] = []
    hist[">20 Hits"] = []
    list_hits_per_mc = dic["count_hits"]
    hist["All"] = list_hits_per_mc
    
    if args == "All": #return all hits
        return hist
    
    pdg = dic["pdg"] #separate hits by pdg
    if args == "photon" or args == "photonSec" or args == "": #return only the hits that have a pdg photon
        # hist["Only Photons"] = []
        for i in range(0, len(list_hits_per_mc)):
            if pdg[i] == 22:
                hist["Only Photons"].append(list_hits_per_mc[i])
        
        if args == "photonSec" or args == "":
            # hist["Only Photons Produced by Secondary Particles"] = []
            hits_produced_secondary = dic["hits_produced_secondary"] #doesnt have same shape as list mcs
            hits_mc_produced_secondary = dic["hits_mc_produced_secondary"]
            for i in range(0, len(list_hits_per_mc)): #get all the hits that are photons and produced by secondary particles
                if pdg[i] == 22 and hits_mc_produced_secondary[i]:
                    hist["Only Photons Produced by Secondary Particles"].append(list_hits_per_mc[i])
            
    if args == "neutron" or args == "neutronSec" or args == "": #return only the hits that have a pdg neutron
        # hist["Only Neutrons"] = []
        for i in range(0, len(list_hits_per_mc)):
            if pdg[i] == 2112:
                hist["Only Neutrons"].append(list_hits_per_mc[i])
        
        if args == "neutronSec" or args == "":
            # hist["Only Neutrons Produced by Secondary Particles"] = []
            hits_mc_produced_secondary = dic["hits_mc_produced_secondary"]
            
            for i in range(0, len(list_hits_per_mc)):
                if pdg[i] == 2112 and hits_mc_produced_secondary[i]:
                    hist["Only Neutrons Produced by Secondary Particles"].append(list_hits_per_mc[i])
                    
    if args == "electron" or args == "": #return only the hits that have a pdg electron; also get ones produced by secondary particles
        # hist["Only Electrons"] = []
        for i in range(0, len(list_hits_per_mc)):
            if pdg[i] == 11:
                hist["Only Electrons"].append(list_hits_per_mc[i])
        
        # hist["Only Electrons Produced by Secondary Particles"] = []
        hits_mc_produced_secondary = dic["hits_mc_produced_secondary"]
        for i in range(0, len(list_hits_per_mc)):
            if pdg[i] == 11 and hits_mc_produced_secondary[i]:
                hist["Only Electrons Produced by Secondary Particles"].append(list_hits_per_mc[i])
                    
    if args == "allPDG" or args == "": #return all hits with pdg as key and count as value
        pdg_unique = np.unique(pdg)
        for i in range(0, len(pdg_unique)):
            hist[pdg_unique[i]] = 0
        for i in range(0, len(list_hits_per_mc)):
            hist[pdg[i]] += 1
    
    if args == "multiHits" or args == "": #return all hits with keys of one hit, >1 hit, >5 hits, >10 hits, >20 hits
        # hist["1 Hit"] = []
        # hist[">1 Hits"] = []
        # hist[">5 Hits"] = []
        # hist[">10 Hits"] = []
        # hist[">20 Hits"] = []
        for i in range(0, len(list_hits_per_mc)):
            if list_hits_per_mc[i] == 1:
                hist["1 Hit"].append(list_hits_per_mc[i])
            elif list_hits_per_mc[i] > 1:
                hist[">1 Hits"].append(list_hits_per_mc[i])
            elif list_hits_per_mc[i] > 5:
                hist[">5 Hits"].append(list_hits_per_mc[i])
            elif list_hits_per_mc[i] > 10:
                hist[">10 Hits"].append(list_hits_per_mc[i])
            elif list_hits_per_mc[i] > 20:
                hist[">20 Hits"].append(list_hits_per_mc[i])
            
            
    return hist

def momPerMC(dic, args = "", byPDG: bool = False):
    """
    Given a hit by a particle, what is that particles momentum.
    Inputs: dic from setup,
            args is the type of hits to calculate:
                "" -- calculate all args \n
                "all" -- all particles \n
                "onlyOH" -- all particles with only one hit \n
                    "byPDG" -- if True, will return a dictionary with keys of pdg and values of momentum \n
                "only+H" -- all particles with more than one hit \n
                    "byPDG" -- if True, will return a dictionary with keys of pdg and values of momentum \n
                "onlyParPhoton" -- all particles with a parent photon \n
                "ptBelow10R" -- all particles with a vertex radius below 10 \n
                "multiHits" -- all particles with keys of one hit, >1 hit, >5 hits, >10 hits, >20 hits \n
                "multiHitsExcludeOne" -- all particles with keys of >1 hit, >5 hits, >10 hits, >20 hits \n
    Outputs: hist, dictionary of values (count of hits) with keys: \n
        "All" -- all particles momentum \n
        "onlyOH" -- all particles momentum with only one hit \n
        "only+H" -- all particles momentum with more than one hit \n
        "onlyParPhoton" -- all particles momentum with a parent photon \n
        "ptBelow10R" -- all particles momentum with a vertex radius below 10 \n
        "multiHits" -- all particles with keys of one hit, >1 hit, >5 hits, >10 hits, >20 hits \n
        "multiHitsExcludeOne" -- all particles with keys of >1 hit, >5 hits, >10 hits, >20 hits \n
    """
    # Create the dictionary
    hist = {}
    hist["All"] = []
    hist["onlyOH"] = []
    hist["only+H"] = []
    hist["onlyParPhoton"] = []
    hist["ptBelow10R"] = []
    hist["multiHits"] = []
    hist["multiHitsExcludeOne"] = []
    p = dic["p"]
    # print(f"dic: {dic['p']}")
    
    if byPDG:
        pdg = dic["pdg"]
    
    if args == "" or args == "All": #regular get all momenta
        hist["all"] = p
        return hist

    if args == "onlyOH" or args == "only+H":
        count_hits = dic["count_hits"] #the index of the mcParticle for each hit
        #seperate hits based on if they occur once or more than once with the same mcParticle
        for i in range(0, len(count_hits)):
            if count_hits[i] == 1:
                if byPDG and args == "onlyOH":
                    if pdg[i] in hist:
                        hist[pdg[i]].append(p[i])
                    else:
                        hist[pdg[i]] = [p[i]]
                hist["onlyOH"].append(p[i])
            else:
                if byPDG and args == "only+H":
                    if pdg[i] in hist:
                        hist[pdg[i]].append(p[i])
                    else:
                        hist[pdg[i]] = [p[i]]
                hist["only+H"].append(p[i])
            hist["All"].append(p[i])
        if byPDG:
            hist["All"] = []
        if args == "onlyOH":
            hist["only+H"] = []
        else:
            hist["onlyOH"] = []
        return hist
    
    if args == "onlyParPhoton":
        """
        If the mcParticle has a parent that is a photon, return the momentum of the mcParticle.
        Keyword arguments:
        argument -- description
        """
        photon = dic["has_par_photon"] #list t(1) or false(0) if mcParticle has a parent thats a photon
        for i in range(0, len(photon)):
            if photon[i]:
                hist["onlyParPhoton"].append(p[i])
        return hist
    
    if args == "ptBelow10R":
        """
        If the mcParticle's vertex radius is below 10, fill the histogram with the transverse momentum of the mcParticle.
        Keyword arguments:
        argument -- description
        """
        R = dic["R"]
        px = dic["px"]
        py = dic["py"]
        
        for i in range(0, len(R)):
            if R[i] < 0.01:
                hist["ptBelow10R"].append(math.sqrt(px[i]**2 + py[i]**2))
        return hist
    
    if args == "multiHits" or args == "multiHitsExcludeOne":
        hist["1 Hit"] = []
        hist[">1 Hits"] = []
        hist[">5 Hits"] = []
        hist[">10 Hits"] = []
        hist[">20 Hits"] = []
        count_hits = dic["count_hits"]
        
        for i in range(0, len(count_hits)):
            if count_hits[i] == 1 and not args.endswith("ExcludeOne"):
                hist["1 Hit"].append(p[i])
            if count_hits[i] > 1:
                hist[">1 Hits"].append(p[i])
            if count_hits[i] > 5:
                hist[">5 Hits"].append(p[i])
            if count_hits[i] > 10:
                hist[">10 Hits"].append(p[i])
            if count_hits[i] > 20:
                hist[">20 Hits"].append(p[i])
        return hist       
                
def PDGPerMC(dic, args = "", sepSecondary: bool = False):
    """
    Given a hit, what is the PDG of the particle which produced that hit.
    Inputs: dic from setup,
            args is the type of hits to calculate:
                "" -- calculate all args \n
                "all" -- all particles \n
                "electron" -- all particles with pdg 11 \n
                "gen" -- all particles with generator status 1 (WIP) \n
    Outputs: hist, dictionary of values (count of hits) with keys:
        "all" -- all particles \n
        "electron" -- all particles with pdg 11 \n
        "e_photon_parent" -- all particles with pdg 11 and a parent photon \n
        "allGens" -- all particles with generator status \n
        "primary" -- all particles with generator status 1 \n
        "secondary" -- all particles with generator status non1 \n
    """
    pdg = dic["pdg"]
    hist = {}
    hist["all"] = []
    hist["electron"] = []
    hist["e_photon_parent"] = []
    hist["allGens"] = []
    hist["primary"] = []
    hist["secondary"] = []
    if args == "" or args == "all": ##want to return all pdg's as a dictionary of pdg, key = pdg, value = count
        for i in range(0, len(pdg)):
            if str(pdg[i]) in hist:
                hist[str(pdg[i])] += 1
            else:
                hist[str(pdg[i])] = 1
        return hist
    if args == "electron":
        """
        Given there was a hit, add a count if the pdg is an electron. 
        If the pdg of the particle is an electron AND has a parent photon, add a count to hist["e_photon_parent"].
        Return: hist, dictionary with keys:
            "electron" -- count of electrons
            "e_photon_parent" -- pdg of electrons with a parent photon
        """
        hist["electron"] = 0
        hist["e_photon_parent"] = []
        has_par_photon = dic["has_par_photon"]
        for i in range(0, len(pdg)):
            if pdg[i] == 11:
                hist["electron"] += 1
                if has_par_photon[i]:
                    hist["e_photon_parent"].append(pdg[i])
        return hist
    if args == "gen":
        """
        given generator status, return a dictionary with the following:
        key: allGens, value: all particles
        key: primary, value: particles with generator status 1
        key: secondary, value: particles with generator status 2
        Return: dictionary with keys:
            "allGens" -- all particles
            "primary" -- particles with generator status 1
            "secondary" -- particles with generator status non1
        ...
        """
        gen = dic["gens"]
        hist["allGens"] = 0
        hist["primary"] = 0
        hist["secondary"] = 0
        #gen 0 is created by simulation
        #gen 1 is original particles
        
        for i in range(0, len(gen)):
            hist["all"] += 1
            if gen[i] == 1:
                hist["primary"] += 1
            if gen[i] != 1:
                hist["secondary"] += 1
        return hist
    
def occupancy(dic, args = ""):
    """
    Determines occupancy and related values.
    Inputs: dic from setup,
            args is the type to calculate:
                "" -- calculate all args \n
                "n_cells" -- number of cells fired by an mcParticle \n
                "percentage_fired" -- percentage of cells fired by an mcParticle \n
                "occupancy_per_batch_sum_events" -- average occupancy (%) per batch, summing occupancies an event \n
                "occupancy_per_batch_sum_events_non_normalized" -- average occupancy (not %) per batch, summing occupancies an event \n
                "occupancy_per_batch_sum_batches" -- average occupancy (%) per batch, summing occupancies a batch \n
                "occupancy_per_batch_sum_batches_non_normalized" -- average occupancy (not %) per batch, summing occupancies a batch \n
                "cells_per_layer" -- cells per layer \n
            note most of these occupancy values are pre-calculated in the data file
            
    Outputs: hist, dictionary of values (count of hits) with keys: \n
        "" -- all values \n
        "n_cells" -- number of cells fired by an mcParticle\n
        "percentage_fired" -- percentage of cells fired by an mcParticle\n
        "occupancy_per_batch_sum_events" -- average occupancy (%) per batch, summing occupancies an event\n
        "occupancy_per_batch_sum_events_error" -- average occupancy (%) per batch, summing occupancies an event\n
        "occupancy_per_batch_sum_events_non_normalized" -- average occupancy (not %) per batch, summing occupancies an event\n
        "occupancy_per_batch_sum_events_non_normalized_error" -- average occupancy (not %) per batch, summing occupancies an event\n
        "occupancy_per_batch_sum_batches" -- average occupancy (%) per batch, summing occupancies a batch\n
        "occupancy_per_batch_sum_batches_error" -- average occupancy (%) per batch, summing occupancies a batch\n
        "occupancy_per_batch_sum_batches_non_normalized" -- average occupancy (not %) per batch, summing occupancies a batch\n
        "occupancy_per_batch_sum_batches_non_normalized_error" -- average occupancy (not %) per batch, summing occupancies a batch\n
        "n_cells_per_layer" -- cells per layer\n
        "total_number_of_cells" -- total number of cells\n
        "total_number_of_layers" -- total number of layers\n
    """
    hist = {}
    hist["n_cells"]= []
    hist["percentage_fired"] = []
    hist["n_cells_per_layer"] = []
    hist["total_number_of_cells"] = []
    hist["total_number_of_layers"] = []
    
    hist["occupancy_per_batch_sum_events"] = []
    hist["occupancy_per_batch_sum_events_error"] = []
    hist["occupancy_per_batch_sum_events_non_normalized"] = []
    hist["occupancy_per_batch_sum_events_non_normalized_error"] = []
    
    hist["occupancy_per_batch_sum_batches"] = []
    hist["occupancy_per_batch_sum_batches_error"] = []
    
    
    if args == "n_cells" or args == "":
        hist["n_cells"] = dic["list_n_cells_fired_mc"]
        
    if args == "percentage_fired" or args == "":
        hist["percentage_fired"] = dic["percentage_of_fired_cells"]
        
    if args == "cells_per_layer" or args == "":
        hist["n_cells_per_layer"] = dic["n_cell_per_layer"]
        hist["total_number_of_cells"] = dic["total_number_of_cells"]
    hist["total_number_of_layers"] = dic["total_number_of_layers"]
    
    if args == "occupancy_per_batch_sum_events" or args == "":
        hist["occupancy_per_batch_sum_events"] = dic["occupancy_per_batch_sum_events"]
        hist["occupancy_per_batch_sum_events_error"] = dic["occupancy_per_batch_sum_events_error"]
        
    if args == "occupancy_per_batch_sum_events_non_normalized" or args == "":
        hist["occupancy_per_batch_sum_events_non_normalized"] = dic["occupancy_per_batch_sum_events_non_normalized"]
        hist["occupancy_per_batch_sum_events_non_normalized_error"] = dic["occupancy_per_batch_sum_events_non_normalized_error"]
    
    if args == "occupancy_per_batch_sum_batches" or args == "":
        hist["occupancy_per_batch_sum_batches"] = dic["occupancy_per_batch_sum_batches"]
        hist["occupancy_per_batch_sum_batches_error"] = dic["occupancy_per_batch_sum_batches_error"]
        
    if args == "occupancy_per_batch_sum_batches_non_normalized" or args == "":
        hist["occupancy_per_batch_sum_batches_non_normalized"] = dic["occupancy_per_batch_sum_batches_non_normalized"]
        hist["occupancy_per_batch_sum_batches_non_normalized_error"] = dic["occupancy_per_batch_sum_batches_non_normalized_error"]
    
    return hist





def plotMomentum(args=""):
    """
    Plot the momentum of all particles.
    
    Inputs:
        args -- the type of momentum to plot
            "" -- plot all momentums plots \n
            "momentum-all" -- plot all momentums \n
            "momentum-onlyOH" -- plot only momentums with one hit \n
            "momentum-only+H" -- plot only momentums with more than one hit \n
            "momentum-onlyParPhoton" -- plot only momentums with a parent photon \n
            "momentum-ptBelow10R" -- plot only momentums with a vertex radius below 10 \n
            "momentum-onlyOH-pdg" -- plot only momentums with one hit separated by pdg \n
            "momentum-only+H-pdg" -- plot only momentums with more than one hit separated by pdg \n
            "momentum-multiHits" -- plot only momentums with keys of 1 hit, >1 hit, >5 hits, >10 hits, >20 hits \n
            "momentum-multiHitsExcludeOne" -- plot only momentums with keys of >1 hit, >5 hits, >10 hits, >20 hits \n
    Returns: no return, just saves the plots
    """
    if args == "momentum-all" or args == "":
        hist = momPerMC(dic, "")
        hist_plot(hist['All'], imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + ".png", "Momentum of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles")
        hist_plot(hist['All'], imageOutputPath + "momentumMC" + str(numFiles) + "Loggedx.png", "Momentum of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
        hist_plot(hist['All'], imageOutputPath + "momentumMC" + str(numFiles) + "Logged.png", "Momentum of " + str(typeFile) + "MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True, logY=True)

    if args == "momentum-onlyOH" or args == "":
        hist = momPerMC(dic, "onlyOH")
        hist_plot(hist["onlyOH"], imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "onlyOneHit.png", "Momentum of " + str(typeFile) + " MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles")
        hist_plot(hist["onlyOH"], imageOutputPath + "momentum" + str(typeFile) + " MC" + str(numFiles) + "onlyOneHitLoggedx.png", "Momentum of " + str(typeFile) + " MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
        hist_plot(hist["onlyOH"], imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "onlyOneHitLogged.png", "Momentum of " + str(typeFile) + " MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)

    if args == "momentum-only+H" or args == "":
        hist = momPerMC(dic, "only+H")
        hist_plot(hist["only+H"], imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "only+Hit.png", "Momentum of " + str(typeFile) + " MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles",
                    xMin=0, xMax=0.3)
        hist_plot(hist["only+H"], imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "only+HitLoggedx.png", "Momentum of " + str(typeFile) + " MC particles \nwith more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
        hist_plot(hist["only+H"], imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "only+HitLogged.png", "Momentum of " + str(typeFile) + " MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)
    
    if args == "momentum-onlyParPhoton" or args == "":
        hist = momPerMC(dic, "onlyParPhoton")
        hist_plot(hist, imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "onlyParentPhoton.png", "Momentum of " + str(typeFile) + " MC particles with a Parent Photon (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles")
        hist_plot(hist, imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "onlyParentPhotonLoggedx.png", "Momentum of " + str(typeFile) + " MC particles with a Parent Photon (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
        hist_plot(hist, imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "onlyParentPhotonLogged.png", "Momentum of " + str(typeFile) + " MC particles with a Parent Photon (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)
    
    if args == "momentum-ptBelow10R" or args == "":
        hist = momPerMC(dic, "ptBelow10R")
        hist_plot(hist, imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "pt10R.png", "Transverse Momentum of " + str(typeFile) + " MC particles within a radius of 10mm (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles")
        hist_plot(hist, imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "pt10RLoggedx.png", "Transverse Momentum of " + str(typeFile) + " MC particles within a radius of 10mm (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
        hist_plot(hist, imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "pt10RLogged.png", "Transverse Momentum of " + str(typeFile) + " MC particles within a radius of 10mm (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)
    
    if args == "momentum-onlyOH":
        hist = momPerMC(dic, "onlyOH")
        hist_plot(hist["onlyOH"], imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "onlyOneHit.png", "Momentum of " + str(typeFile) + " MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles")
        
    if args == "momentum-only+H" or args == "":
        hist = momPerMC(dic, "only+H")
        hist_plot(hist["only+H"], imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "only+Hit.png", "Momentum of " + str(typeFile) + " MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles",
                    xMin=0, xMax=0.3)
        
    if args == "momentum-onlyOH-pdg" or args == "":
        hist = momPerMC(dic, "onlyOH", byPDG=True)
        multi_hist_plot(hist, imageOutputPath + "momentum"  + str(typeFile) + "MC" + str(numFiles) + "onlyOneHitSepPDG.png", "Momentum of "  + str(typeFile) + " MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", barType="step")
        multi_hist_plot(hist, imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "onlyOneHitSepPDGLoggedx.png", "Momentum of " + str(typeFile) + " MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True, barType="step", contrast=True)
        multi_hist_plot(hist, imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "onlyOneHitSepPDGLogged.png", "Momentum of " + str(typeFile) + " MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True, barType="step")
    
    if args == "momentum-only+H-pdg" or args == "":
        hist = momPerMC(dic, "only+H", byPDG=True)
        multi_hist_plot(hist, imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "only+HitSepPDG.png", "Momentum of " + str(typeFile) + " MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles",  xMin=0, xMax=0.3, barType="step")
        multi_hist_plot(hist, imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "only+HitSepPDGLoggedx.png", "Momentum of " + str(typeFile) + " MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True, barType="step")
        multi_hist_plot(hist, imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "only+HitSepPDGLogged.png", "Momentum of " + str(typeFile) + " MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True, barType="step")
    
    if args == "momentum-multiHitsExcludeOne" or args == "":
        hist = momPerMC(dic, "multiHitsExcludeOne")
        multi_hist_plot(hist, imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "MultiHitsExcludeOneLoggedx.png", "Momentum of " + str(typeFile) + " MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
    
    if args == "momentum-multiHits" or args == "":
        hist = momPerMC(dic, "multiHits")
        multi_hist_plot(hist, imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + "MultiHitsLoggedx.png", "Momentum of " + str(typeFile) + " MC particles with hits (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
    
def plotPDG(args=""):
    """
    Plot the PDG of all particles.
    
    Inputs:
        args -- the type of PDG to plot
            "" -- plot all PDGs plots \n
            "pdg-all" -- plot all PDGs \n
            "pdg-electron" -- plot only electrons \n
            "pdg-gen" -- plot only generator status \n
    Returns: no return, just saves the plots
    """
    if args == "pdg-all" or args == "":
        hist = PDGPerMC(dic, "")
        bar_plot(hist.keys(), hist.values(), imageOutputPath + "pdg" + str(typeFile) + "MC" + str(numFiles) + ".png", "PDG of " + str(typeFile) + " MC particles \n(" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", width=1.3, rotation=90, includeLegend=False)
    
    if args == "pdg-electron" or args == "":
        hist = PDGPerMC(dic, "electron")
        bar_plot(["electron"], hist["all"], imageOutputPath + "pdgElectronPhoton" + str(typeFile) + "MC" + str(numFiles) + ".png", "PDG of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80, save=False, label="All Electrons")
        bar_plot(["electron"], hist["electron"], imageOutputPath + "pdgElectronPhoton" + str(typeFile) + "MC" + str(numFiles) + ".png", "PDG of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80, save=False, label="All Electrons")
        bar_plot(["electron"], hist["e_photon_parent"], imageOutputPath + "pdgElectronPhoton" + str(typeFile) + "MC" + str(numFiles) + ".png", "PDG of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80, label="Only Electrons with a Parent Photon")
    
    if args == "pdg-gen" or args == "":
        hist = PDGPerMC(dic, "gen")
        bar_plot("MC Particles", hist["all"], imageOutputPath + "pdgGeneratorStatus" + str(typeFile) + "MC" + str(numFiles) + ".png", "Primary or Secondary " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80, save=False, label="All Particles")
        multi_bar_plot("MC Particles", hist, imageOutputPath + "pdgElectronGeneratorStatus" + str(typeFile) + "MC" + str(numFiles) + ".png", "Primary or Secondary for Electron " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles")

def plotHits(args=""):
    """
    Plot the number of hits of all particles.
    
    Inputs:
        args -- the type of hits to plot
            "" -- plot all hits plots \n
            "hits-all" -- plot all hits \n
            "hits-neutron" -- plot only neutron hits \n
            "hits-photon" -- plot only photon hits \n
            "hits-electron" -- plot only electron hits \n
    Returns: no return, just saves the plots
    """
    if args == "hits-all" or args == "":
        hist = hitsPerMC(dic, "")
        hist_plot(hist["all"], imageOutputPath + "hits" + str(typeFile) + "MC" + str(numFiles) + ".png", "Hits of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Number of MC particles", label="All Particles", autoBin=False, logX=True, logY=True)
    
    if args == "hits-neutron" or args == "":
        hist = hitsPerMC(dic, "neutronSec")
        hist_plot(hist["all"], imageOutputPath + "hits" + str(typeFile) + "MC" + str(numFiles) + ".png", "Hits of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Number of MC particles", label="All Particles", autoBin=False, logX=True, logY=True)
        multi_hist_plot(hist, imageOutputPath + "hitsNeutron" + str(typeFile) + "MC" + str(numFiles) + ".png", "Hits of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Number of MC particles", label="All Particles", barType="step", autoBin=False, binLow=0.1, binHigh=9000, binSteps=5, binType="lin", logY=True, contrast=True)
        multi_hist_plot(hist, imageOutputPath + "hitsNeutronZoomed" + str(typeFile) + "MC" + str(numFiles) + ".png", "Hits of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Number of MC particles", label="All Particles", barType="step", autoBin=False, binLow=0.1, binHigh=900, binSteps=5, binType="lin", logY=True)
    
    if args == "hits-photon" or args == "":
        hist=hitsPerMC(dic, "photonSec")
        multi_hist_plot(hist, imageOutputPath + "hitsPhoton" + str(typeFile) + "MC" + str(numFiles) + ".png", "Hits of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Number of MC particles", label="All Particles", barType="step", autoBin=False, binLow=0.1, binHigh=9000, binSteps=5, binType="lin", logY=True, contrast=True)
        multi_hist_plot(hist, imageOutputPath + "hitsPhotonZoomed" + str(typeFile) + "MC" + str(numFiles) + ".png", "Hits of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Number of MC particles", label="All Particles", barType="step", autoBin=False, binLow=0.1, binHigh=900, binSteps=5, binType="lin", logY=True)
    
    if args == "hits-electron" or args == "":
        hist=hitsPerMC(dic, "electron")
        multi_hist_plot(hist, imageOutputPath + "hitsElectron" + str(typeFile) + "MC" + str(numFiles) + ".png", "Hits of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Number of MC particles", label="All Particles", barType="step", autoBin=False, binLow=0.1, binHigh=9000, binSteps=5, binType="lin", logY=True, contrast=True)
        multi_hist_plot(hist, imageOutputPath + "hitsElectronZoomed" + str(typeFile) + "MC" + str(numFiles) + ".png", "Hits of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Number of MC particles", label="All Particles", barType="step", autoBin=False, binLow=0.1, binHigh=900, binSteps=5, binType="lin", logY=True)

def plotHitsOverlay(args=""):
    """
    Plot the overlay of the background and signal hits.
    
    Inputs:
        args -- the type of hits to plot
            "" -- plot all hits plots \n
            "hitsOverlay-all" -- plot all hits \n
            "hitsOverlay-electron" -- plot only electron hits \n
            "hitsOverlay-photon" -- plot only photon hits \n
    Returns: no return, just saves the plots
    """
    if args == "hitsOverlay-all" or args == "":
        histSignal = hitsPerMC(dic, "all")
        histBkg = hitsPerMC(dicbkg, "all")
        histBkg["All Particles BKG"] = histBkg.pop("All")
        histBkg["All Particles Signal"] = histSignal["All"]
        multi_hist_plot(histBkg, imageOutputPath + "hitsDensitySignalBkgZoomedMC" + str(numFiles) + ".png", "Hits of BKG and Signal MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Density of MC particles", label="All Particles Signal", autoBin=False, binLow=0.1, binHigh=900, binSteps=5, binType="lin", barType="step", logY=True, density=True)
    
    if args == "hitsOverlay-electron" or args == "":
        histSignal = hitsPerMC(dic, "electron")
        histBkg = hitsPerMC(dicbkg, "electron")
        bar_step_multi_hist_plot(histSignal["Only Electrons"], histBkg, imageOutputPath + "hitsSignalBkgBarStepMC" + str(numFiles) + ".png", "Hits of BKG and Signal MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Number of MC particles", label="All Particles Signal", autoBin=False, binLow=0.1, binHigh=9000, binSteps=5, binType="lin", logY=True)
        histBkg["Electron Particles Signal"] = histSignal["Only Electrons"]
        histBkg["Electron Particles BKG"] = histBkg.pop("Only Electrons")
        histBkg.pop("all")
        multi_hist_plot(histBkg, imageOutputPath + "hitsElectronDensitySignalBkgZoomedMC" + str(numFiles) + ".png", "Hits of BKG and Signal MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Density of MC particles", label="All Particles Signal", autoBin=False, binLow=0.1, binHigh=900, binSteps=5, binType="lin", barType="step",logY=True, density=True)
    
    if args == "hitsOverlay-photon" or args == "":
        histSignal = hitsPerMC(dic, "photonSec")
        histBkg = hitsPerMC(dicbkg, "photonSec")
        histBkg["Photon Particles BKG"] = histBkg.pop("Only Photons")
        histBkg["Photon Particles Signal"] = histSignal["Only Photons"]
        multi_hist_plot(histBkg, imageOutputPath + "hitsPhotonDensitySignalBkgZoomedMC" + str(numFiles) + ".png", "Hits of BKG and Signal MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Density of MC particles", label="All Particles Signal", autoBin=False, binLow=0.1, binHigh=900, binSteps=5, binType="lin", barType="step", logY=True, density=True)

def plotMomOverlay(args=""):
    """
    Plot the overlay of the background and signal momentum.
    
    Inputs:
        args -- the type of momentum to plot
            "" -- plot all momentum plots \n
            "momentumOverlay-all" -- plot all momentums \n
    Returns: no return, just saves the plots
    """
    if args == "momentumOverlay-all" or args == "":
        histSignal = momPerMC(dic, "All")
        histBkg = momPerMC(dicbkg, "All")
        histBkg["All Particles BKG"] = histBkg.pop("All")
        histBkg["All Particles Signal"] = histSignal["All"]
        multi_hist_plot(histBkg, imageOutputPath + "momSignalBkgDensityZoomedMC" + str(numFiles) + ".png", "Momentum of BKG and Signal MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (Gev)", yLabel="Density of MC particles", label="All Particles Signal", autoBin=False, binLow=0.0001, binHigh=50, binSteps=0.3, binType="exp", barType="step",logY=True, logX=True, density=True)

def plotOccupancy(args=""):
    """
    Plot the occupancy of the detector.
    
    Inputs:
        args -- the type of occupancy to plot
            "" -- plot all occupancy plots \n
            "occupancy-nCellsFired" -- plot the number of cells fired \n
            "occupancy-BatchedBatch" -- plot the occupancy per batch \n
            "occupancy-BatchedBatchNN" -- plot the occupancy per batch non normalized \n
            "occupancy-nCellsPerLayer" -- plot the number of cells per layer \n
    Returns: no return, just saves the plots
    """
    #total number of layers: 112
    #total number of cells 56448
    if args == "occupancy-nCellsFired" or args == "":
        hist = occupancy(dic, "n_cells")
        hist_plot(hist["n_cells"], imageOutputPath + "nCellsFired"+str(typeFile)+"MC" + str(numFiles) + ".png", "Number of Cells Fired by Particles (" + str(numFiles) + " Files)", xLabel="Number of cells fired by an MC particle", yLabel="Count", xMin=0, xMax=4000, binLow=0.01, binHigh=4000, binSteps=0.3, binType="lin")
    
    # hist_plot(hist["percentage_fired"], imageOutputPath + "occupancyPercMC" + str(numFiles) + ".png", "Occupancy of the detector (" + str(numFiles) + " Files)", xLabel="Percentage of cells fired", yLabel="Count MC particles", xMin=0, xMax=80, binHigh=80, binSteps=1, binType="lin")
    
    layers = [i for i in range(0, hist["total_number_of_layers"])]
    batch = ""
    if typeFile == "Bkg":
        batch = "20 BKG File Batch"
    elif typeFile == "Signal":
        batch = "1 Signal File Batch"
        
    if args == "occupancy-BatchedBatch" or args == "":
        hist = occupancy(dic, "occupancy_per_batch_sum_batches")
        xy_plot(layers, hist["occupancy_per_batch_sum_batches"], imageOutputPath + "occupancy"+str(typeFile)+"FileBatchMC" + str(numFiles) + ".png",
                "Average Occupancy Across Each " + batch + " (" + str(numFiles) + " Files)",
                xLabel="Radial Layer Index", yLabel="Average Channel Occupancy [%]",
                includeLegend=False, label="", scatter=True, errorBars=True, yerr = hist["occupancy_per_batch_sum_batches_error"])
    
    if args == "occupancy-BatchedBatchNN" or args == "":
        hist = occupancy(dic, "occupancy_per_batch_sum_batches_non_normalized")
        xy_plot(layers, hist["occupancy_per_batch_file_non_normalized"], imageOutputPath + "occupancy"+str(typeFile)+"FileBatchNNMC" + str(numFiles) + ".png", 
                "Average Occupancy Across " + batch + " Non Normalized (" + str(numFiles) + " Files)",
                xLabel="Radial Layer Index", yLabel="Average Channel Occupancy",
                includeLegend=False, label="", scatter=True, errorBars=True, yerr = hist["occupancy_per_batch_file_non_normalized_error"])
    
    if args == "occupancy-nCellsPerLayer" or args == "":
        x = [int(i) for i in list(hist["n_cells_per_layer"].keys())]
        xy_plot(x, list(hist["n_cells_per_layer"].values()), imageOutputPath + "nCellsPerLayer"+str(typeFile)+"MC" + str(numFiles) + ".png", "Cells Per Layer (" + str(numFiles) + " Files)", xLabel="Layer Number", yLabel="Cells Per Layer", includeLegend=False, label="")
    
def plotWireChamber(args=""):
    """
    Plot the wire chamber.
    
    Inputs:
        args -- the type of wire chamber to plot
            "" -- plot all wire chamber plots \n
            "wireChamber-all" -- plot all wire chamber \n
    """
    if args == "wireChamber-all" or args == "":  
        hist = occupancy(dic, "cells_per_layer")
        plot_wire_chamber(hist["total_number_of_layers"], hist["n_cells_per_layer"], imageOutputPath + "wireChamberFirstQuad" + ".png", title="", firstQuadrant=True)
    
def plotAll(args=""):
    """
    Plot all histograms.
    """
    ###momentum plots
    plotMomentum()
    
    ###pdg plots
    plotPDG()
    
    ###hits plots
    plotHits()
    
    ###overlay bkg and signal mom
    plotMomOverlay()
    
    ###overlay bkg and signal hits
    plotHitsOverlay()
    
    ###occupancy
    plotOccupancy()
    
    ###wireChamber
    plotWireChamber()
    
def genPlot(inputArgs):
    """
    Used by the argparse to generate the desired plot(s). 
    Basically just want to map the inputArgs to the correct function.
    Inputs: inputArgs, should be one argument either "" or from typePlots
            imageOutputPath, path to save the image
    Outputs: plot saved to imageOutputPath
    """
    if len(inputArgs) != 1 or len(inputArgs) != 2:
        raise argparse.ArgumentTypeError("The --plot argument requires max 2 arguments, the type of plot to generate, the type of files to use.")
    
    typePlot = inputArgs[0]
    typeFile = inputArgs[1]
    setup(typeFile)
    # Mapping strings to functions
    function_map = {
        "": plotAll,
        "all": plotAll,
        "momentum-all": plotMomentum,
        "momentum-onlyOH": plotMomentum,
        "momentum-only+H": plotMomentum,
        "momentum-onlyParPhoton": plotMomentum,
        "momentum-ptBelow10R": plotMomentum,
        "momentum-onlyOH-pdg": plotMomentum,
        "momentum-only+H-pdg": plotMomentum,
        "momentum-multiHits": plotMomentum,
        "momentum-multiHitsExcludeOne": plotMomentum,
        "pdg-all": plotPDG,
        "pdg-electron": plotPDG,
        "pdg-gen": plotPDG,
        "hits-all": plotHits,
        "hits-neutron": plotHits,
        "hits-photon": plotHits,
        "hits-electron": plotHits,
        "hitsOverlay-all": plotHitsOverlay,
        "hitsOverlay-electron": plotHitsOverlay,
        "hitsOverlay-photon": plotHitsOverlay,
        "momentumOverlay-all": plotMomOverlay,
        "occupancy-nCellsFired": plotOccupancy,
        "occupancy-BatchedBatch": plotOccupancy,
        "occupancy-BatchedBatchNN": plotOccupancy,
        "occupancy-nCellsPerLayer": plotOccupancy,
        "wireChamber-all": plotWireChamber
    }

    # Execute the corresponding function if the key exists
    if inputArgs in function_map:
        function_map[typePlot](typePlot)  # Calls func_b()
    else:
        print("No match found")
          
       
#'''
#create argument parser so someone can create plots without hard coding
parser = argparse.ArgumentParser()
typePlots = ["all", 
             "momentum-all", "momentum-onlyOH", "momentum-only+H", 
             "momentum-onlyParPhoton", "momentum-ptBelow10R", 
             "pdg-all", "pdg-electron", "pdg-gen", "hitsPerMC", "occupancy"]
parser.add_argument('--plot', help="Plot histogram \n-- plotType(str): " +
                    str(typePlots) + "\n-- fileType(str): [Bkg], [Signal], [Combined]", type=str, default="", nargs='+')
args = parser.parse_args()

if args.plot and args.plot != "":
    try:
        print(f"Parsed --plot arguments: ...")
        genPlot(args.plot)
    except ValueError as e:
        parser.error(str(e))
#'''
'''    
example (plot all):
/work/submit/poulin/fccproject-tracking/detector_beam_backgrounds/tracking/trBkgPlt.py --plot

example(plot momentum)
/work/submit/poulin/fccproject-tracking/detector_beam_backgrounds/tracking/trBkgPlt.py --plot momentum-all
'''