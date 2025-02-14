#Alexander Poulin Jan 2025
import ROOT
import numpy as np 
from utilities.functions import hist_plot, multi_hist_plot, \
    bar_plot, multi_bar_plot, xy_plot, bar_step_multi_hist_plot, heatmap
from utilities.pltWireCh import plot_wire_chamber
import argparse
import sys
import matplotlib.pyplot as plt
import math
import random

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
    In general, I split the functions into two parts: one that calculates the values and one that plots the values.
See bottom for example usage and documentation of argument parsing.

To run, use the following command:
python <'path'> <'function'> <'fileType'>
    where:  <'path'> is the path to this file
            <'function'> is the function to run
            <'fileType'> is the type of file to run the function on
All functions:
    typePlots = ["", "all", 
                "momentum-all", "momentum-onlyOH", "momentum-only+H",
                    "momentum-onlyParPhoton", "momentum-ptBelow10R", "momentum-onlyOH-pdg",
                    "momentum-only+H-pdg", "momentum-multiHits", "momentum-multiHitsExcludeOne",
                    "pdg-all", "pdg-electron", "pdg-gen",
                    "hits-all", "hits-neutron", "hits-photon", "hits-electron",
                    "hitsOverlay-all", "hitsOverlay-electron", "hitsOverlay-photon",
                    "momentumOverlay-all",
                    "occupancy-nCellsFired", "occupancy-BatchedBatch", "occupancy-BatchedBatchNN",
                    "occupancy-nCellsPerLayer",
                    "wireChamber-all",
                    "plot3dPos", "hitRadius-all", "hitRadius-layers-radius", "hitRadius-all-layers"
                ]
"""



available_functions = ["hitsPerMC", "momPerMC", "PDGPerMC", "occupancy", "hitRadius", "plot3dPosition"]
dic = {}
dicbkg = {}
dicBkgnumFiles = 500
# imageOutputPath = "fccproject-tracking/detector_beam_backgrounds/tracking/images/test/" #mit-submit
imageOutputPath = "public/work/fccproject-tracking/detector_beam_backgrounds/tracking/images/lxplus/" #lxplus
numFiles = 0


#change to personal directories in here:
def setup(typefile: str ="Bkg", includeBkg: bool =False, numfiles=500):
    """
    Setups the file paths and outputs.
    Data paths will lead to either the background or signal data or their combined files (not yet tested).
    Output paths should be entirely dependent on your directory and where you want to specify.
    
    Note data paths should point to a .npy file that contains a dictionary with the following keys: \n
        "hits" -- list of hits per mcParticle \n
        "pdg" -- list of pdg per mcParticle \n
        "p" -- list of momentum per mcParticle \n
        "pt" -- list of pt per mcParticle \n
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
    # backgroundDataPath = "fccproject-tracking/detector_beam_backgrounds/tracking/data/bkg_background_particles_"+str(numFiles)+".npy" #mit-submit
    # backgroundDataPath = "fccproject-tracking/detector_beam_backgrounds/tracking/data/occupancy_tinker/bkg_background_particles_"+str(numFiles)+".npy" #mit-submit
    backgroundDataPath = "public/work/fccproject-tracking/detector_beam_backgrounds/tracking/data/lxplusData/bkg_background_particles_" + str(numfiles) + "_v6_R1_PN.npy" #lxplus
    dicbkgDataPath = "public/work/fccproject-tracking/detector_beam_backgrounds/tracking/data/lxplusData/bkg_background_particles_" + str(dicBkgnumFiles) + "_v6_R4_P1.npy" #lxplus
    combinedDataPath = "fccproject-tracking/detector_beam_backgrounds/tracking/data/combined/"
    # signalDataPath = "fccproject-tracking/detector_beam_backgrounds/tracking/data/occupancy_tinker/signal_background_particles_"+str(numFiles)+".npy" #mit-submit
    signalDataPath = "public/work/fccproject-tracking/detector_beam_backgrounds/tracking/data/lxplusData/signal_background_particles_" + str(numfiles) + "_v6_R1_PN.npy" #lxplus
    
    if typefile == "Bkg":
        dic = np.load(backgroundDataPath, allow_pickle=True).item()
        print(f"Reading dictionary from: {backgroundDataPath}")
    elif typefile == "Combined":
        dic = np.load(combinedDataPath, allow_pickle=True).item()
        print(f"Reading dictionary from: {combinedDataPath}")
    elif typefile == "Signal":
        dic = np.load(signalDataPath, allow_pickle=True).item()
        print(f"Reading dictionary from: {signalDataPath}")
    else:
        print("Type must be either Bkg, Combined, or Signal")
        sys.exit()
    if includeBkg:
        dicbkg = np.load(dicbkgDataPath, allow_pickle=True).item()
        print(f"Reading extra bkg dictionary from: {dicbkgDataPath}")
    else:
        dicbkg = {}
    global typeFile
    typeFile = typefile
    global numFiles
    numFiles = numfiles
    print(f"Setup complete for {typeFile} data")
    return dic, dicbkg

# dic, dicbkg = setup(typeFile, includeBkg)

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

def momPerMC(dic, args = "", pPtArg="p", byPDG: bool = False):
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
    hist["all"] = []
    hist["onlyOH"] = []
    hist["only+H"] = []
    hist["onlyParPhoton"] = []
    hist["ptBelow10R"] = []
    hist["multiHits"] = []
    hist["multiHitsExcludeOne"] = []
    
    if pPtArg == "p":
        p = dic["p"]
    elif pPtArg == "pt":
        p = dic["pt"]
    else:
        print("pPtarg must be either 'p' or 'pt'")
        sys.exit()
    # print(f"dic: {dic['p']}")
    
    if byPDG:
        pdg = dic["pdg"]
    
    if args == "" or args == "all": #regular get all momenta
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
                "occupancy_per_batch_sum_batches" -- average occupancy (%) per batch, summing occupancies a batch \n
                "occupancy_per_batch_sum_batches_non_normalized" -- average occupancy (not %) per batch, summing occupancies a batch \n
                "cells_per_layer" -- cells per layer \n
                "avg_energy_deposit" -- average energy deposit per batch \n
                "energy_deposit" -- energy deposit per batch \n
            note most of these occupancy values are pre-calculated in the data file
            
    Outputs: hist, dictionary of values (count of hits) with keys: \n
        "" -- all values \n
        "n_cells" -- number of cells fired by an mcParticle\n
        "percentage_fired" -- percentage of cells fired by an mcParticle\n
        "occupancy_per_batch_sum_batches" -- average occupancy (%) per batch, summing occupancies a batch\n
        "occupancy_per_batch_sum_batches_error" -- average occupancy (%) per batch, summing occupancies a batch\n
        "occupancy_per_batch_sum_batches_non_normalized" -- average occupancy (not %) per batch, summing occupancies a batch\n
        "occupancy_per_batch_sum_batches_non_normalized_error" -- average occupancy (not %) per batch, summing occupancies a batch\n
        "n_cells_per_layer" -- cells per layer\n
        "total_number_of_cells" -- total number of cells\n
        "total_number_of_layers" -- total number of layers\n
        "occupancy_per_batch_sum_batches_energy_dep" -- energy deposit per batch\n
        "occupancy_per_batch_sum_batch_avg_energy_dep" -- average energy deposit per batch\n
        "occupancy_per_batch_sum_batch_avg_energy_dep_error" -- average energy deposit per batch\n
    """
    hist = {}
    hist["n_cells"]= []
    hist["percentage_fired"] = []
    hist["n_cells_per_layer"] = []
    hist["total_number_of_cells"] = []
    hist["total_number_of_layers"] = []
    
    hist["occupancy_per_batch_sum_batches"] = []
    hist["occupancy_per_batch_sum_batches_error"] = []
    
    hist["occupancy_per_batch_sum_batches_only_neighbor"] = []
    hist["occupancy_per_batch_sum_batches_only_neighbor_error"] = []
    
    hist["occupancy_per_batch_sum_batches_energy_dep"] = []
    hist["occupancy_per_batch_sum_batch_avg_energy_dep"] = []
    hist["occupancy_per_batch_sum_batch_avg_energy_dep_error"] = []
    
    
    if args == "n_cells" or args == "":
        hist["n_cells"] = dic["list_n_cells_fired_mc"]
        
    if args == "percentage_fired" or args == "":
        hist["percentage_fired"] = dic["percentage_of_fired_cells"]
        
    if args == "cells_per_layer" or args == "":
        hist["n_cells_per_layer"] = dic["n_cell_per_layer"]
        hist["total_number_of_cells"] = dic["total_number_of_cells"]
    hist["total_number_of_layers"] = dic["total_number_of_layers"]
    
    if args == "occupancy_per_batch_sum_batches" or args == "":
        hist["occupancy_per_batch_sum_batches"] = dic["occupancy_per_batch_sum_batches"]
        hist["occupancy_per_batch_sum_batches_error"] = dic["occupancy_per_batch_sum_batches_error"]
        hist["occupancy_per_batch_sum_batches_non_meaned"] = dic["occupancy_per_batch_sum_batches_non_meaned"]
        
    if args == "occupancy_per_batch_sum_batches_non_normalized" or args == "":
        hist["occupancy_per_batch_sum_batches_non_normalized"] = dic["occupancy_per_batch_sum_batches_non_normalized"]
        hist["occupancy_per_batch_sum_batches_non_normalized_error"] = dic["occupancy_per_batch_sum_batches_non_normalized_error"]
    
    if args == "occupancy_per_batch_sum_batches_only_neighbor" or args == "":
        hist["occupancy_per_batch_sum_batches_only_neighbor"] = dic["occupancy_per_batch_sum_batches_only_neighbor"]
        hist["occupancy_per_batch_sum_batches_only_neighbor_error"] = dic["occupancy_per_batch_sum_batches_only_neighbor_error"]
        
    if args == "avg_energy_deposit" or args == "":
        hist["occupancy_per_batch_sum_batch_avg_energy_dep"] = dic["occupancy_per_batch_sum_batch_avg_energy_dep"]
        hist["occupancy_per_batch_sum_batch_avg_energy_dep_error"] = dic["occupancy_per_batch_sum_batch_avg_energy_dep_error"]
        
    if args == "energy_deposit" or args == "":
        edepdic = dic["dic_occupancy_per_batch_sum_batches_energy_dep"] #this is dictionary where key is tuple (unique_layer_index, nphi) and value is energy deposit
        #we want to restructure this so that we have a total layers by nphi matrix where each element is the energy deposit
        #we can then plot this as a heatmap
        total_layers = dic["total_number_of_layers"]
        total_cells = dic["total_number_of_cells"]
        total_nphi = dic['max_n_cell_per_layer']
        print(f"total_layers: {total_layers}, total_cells: {total_cells}, nphi: {dic['max_n_cell_per_layer']}")
        hist["occupancy_per_batch_sum_batches_energy_dep"] = np.zeros((total_layers, total_nphi))
        for key, value in edepdic.items():
            hist["occupancy_per_batch_sum_batches_energy_dep"][key[0], key[1]] = value
    
    return hist

def hitRadius(dic, args = ""):
    """
    Given a hit, what is the radius of the hit.

    Args:
        dic (dictionary): _description_
        args (str, optional): _description_. Defaults to "".

    Returns:
        dictionary: dictionary with keys: \n
            "All" -- a list of all hit's radius \n
    """
    hist = {}
    hist["All"] = []
    hist["Position of Hit"] = []
    hist["Position of Vertex"] = []
    hist["OverlayLayer"] = []
    
    # list_hits_per_mc = dic["count_hits"]
    # list_r_per_mc = dic["R"]
    # list_hits = dic["hits"]
    
    dic_pos = dic["pos_hit"] 
    #this is a list of dictionaries (each is dictionary is an event)
    #each dictionary has a key of mcParticle index and a value of a list of tuples (3position) of where it hit
    #we dont care by what particle generated the hit, but get the number of hits for a certain radius
    
    if args == "All" or args == "":
        for event in dic_pos:
            for mcParticle, positions in event.items():
                for position in positions:
                    # hist["All"].append(math.sqrt(position[0]**2 + position[1]**2 + position[2]**2))
                    hist["All"].append(math.sqrt(position[0]**2 + position[1]**2))
    
    if args == "OverlayLayer" or args == "":
        dic_n_cells_per_layer = dic["n_cell_per_layer"] #just get the number of cells per layer
        total_number_of_layers = dic["total_number_of_layers"]
        hist["OverlayLayer"] = dic_n_cells_per_layer #dictionary where keys are layer number, values are ncells
    return hist


def plot3dPosition(dic, dicbkg, args=""):
    plt.ion()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Generate a color map for particles
    color_map = {}
    
    particle_data_hit = dic["pos_hit"]
    particle_data_ver = dic["pos_ver"]
    # print(f"particle_data: {particle_data[0]}")
    #so particle data is a list of dictionaries
    #there should be a dictionary for every event
    #in each dictionary, the key is the mcParticle index, and the value is a list of positions

    # Iterate through the list of dictionaries
    print(len(particle_data_ver))
    print(len(particle_data_hit)) #these should be the same length
    
    #now we want to match indexes and append the hits onto the vers
    merged_pos = []

    for dict1, dict2 in zip(particle_data_ver, particle_data_hit):
        merged_dict = {}
        
        # Merge first dictionary
        for key, value in dict1.items(): #should only be one tuple that gets added
            # print(f"valuedic1: {value}")
            if key not in merged_dict:
                merged_dict[key] = []
            merged_dict[key].append(value)
        
        # Merge second dictionary
        for key, value in dict2.items(): #a range of possible tuples
            # print(f"valuedic2: {value}")
            if key not in merged_dict:
                print("key not found for hits???")
                input("press Enter to continue...")
                merged_dict[key] = []
            # merged_dict[key].append(value)
            merged_dict[key] += value
        
        merged_pos.append(merged_dict)
        
    # print(f"merged_pos: {merged_pos[0]}")
    # print(f"merged_pos: {len(merged_pos)}")
    
    numEventsCutoff = 0 #set to -1 to get all events (after event 'n', stop)
    numParticlesCutoff = -1 #set to -1 to get all particles
    particleSeen = 0
    for i, file_dict in enumerate(merged_pos): #particle_dict is the dictionary for each event
        # print(f"file_dict: {i}")
        #signal theres 5000 events so lets restrict to 20
        if i != -1 and i > numEventsCutoff:
            print("break numEvents")
            break
        for particle_index, positions in file_dict.items():
            if particle_index not in color_map:
                if numParticlesCutoff != -1 and particleSeen > numParticlesCutoff:
                    print("break numParticles")
                    break
                particleSeen += 1
                color_map[particle_index] = (random.random(), random.random(), random.random())  # Assign a random color
            
            # Convert positions to numpy arrays for easy plotting
            positions = np.array(positions) #for when list of tuples
            
            x, y, z = positions[:, 0], positions[:, 1], positions[:, 2] #for when list of tuples
            # x, y, z = positions[0], positions[1], positions[2] #for when just tuples

            # Plot the trajectory with lines and points
            ax.plot(x, y, z, marker='o', markersize=3, linestyle='-', alpha=0.7, color=color_map[particle_index], label=f'Particle {particle_index}')

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #make tick labels small
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_title('3D Particle Trajectories ' + str(typeFile) + " seen: " + str(particleSeen) + "MC")

    # Show legend (if particles overlap, the legend might repeat)
    # handles, labels = ax.get_legend_handles_labels()
    # unique_labels = dict(zip(labels, handles))  # Remove duplicates
    # ax.legend(unique_labels.values(), unique_labels.keys())
    fig.savefig(imageOutputPath + "3D" + str(typeFile) + "MCTrajE" + str(numEventsCutoff) + "P" + str(numParticlesCutoff), bbox_inches="tight")
    
    ax.view_init(elev=0, azim=0)  # Adjust elevation and azimuth
    fig.savefig(imageOutputPath + "3D" + str(typeFile) + "MC00TrajE" + str(numEventsCutoff) + "P" + str(numParticlesCutoff), bbox_inches="tight")
    
    #plot a radial sphere to show the detector layers with extreme alpha
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 10 * np.outer(np.cos(u), np.sin(v)) *300
    print(x)
    y = 10 * np.outer(np.sin(u), np.sin(v)) * 300
    z = 10 * np.outer(np.ones(np.size(u)), np.cos(v)) * 300
    ax.plot_surface(x, y, z, color='b', alpha=0.1)
    #make sure plot is square
    # ax.set_box_aspect([1,1,1])
    #Set y limit to be 3000
    ax.set_ylim(-3000, 3000)
    ax.set_zlim(-3000, 3000)
    ax.set_xlim(-3000, 3000)
    # fig.savefig(imageOutputPath + "3D" + str(typeFile) + "MC00TrajSphereE" + str(numEventsCutoff) + "P" + str(numParticlesCutoff), bbox_inches="tight")



def plotMomentum(dic, dicbkg, args=""):
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
    #get first two char of args
    pPtArg = ""
    if args[:2] == "pt":
        pPtArg = "pt"
        args = args[2:]
    #get all but first two char of args
    args = args[2:]
    if args == "momentum-all" or args == "":
        hist = momPerMC(dic, "")
        hist_plot(hist['all'], imageOutputPath + "momentum" + str(typeFile) + "MC" + str(numFiles) + ".png", "Momentum of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles")
        hist_plot(hist['all'], imageOutputPath + "momentumMC" + str(numFiles) + "Loggedx.png", "Momentum of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
        hist_plot(hist['all'], imageOutputPath + "momentumMC" + str(numFiles) + "Logged.png", "Momentum of " + str(typeFile) + "MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True, logY=True)

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
    
def plotPDG(dic, dicbkg, args=""):
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

def plotHits(dic, dicbkg, args=""):
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

def plotHitsOverlay(dic, dicbkg, args=""):
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

def plotMomOverlay(dic, dicbkg, args=""):
    """
    Plot the overlay of the background and signal momentum.
    
    Inputs:
        args -- the type of momentum to plot
            "" -- plot all momentum plots \n
            "momentumOverlay-all" -- plot all momentums \n
            "ptmomentumOverlay-all" -- plot all transverse momentums \n
    Returns: no return, just saves the plots
    """
    if args == "momentumOverlay-all" or args == "":
        histSignal = momPerMC(dic, "all")
        histBkg = momPerMC(dicbkg, "all")
        histBkg["All Particles BKG"] = histBkg.pop("all")
        histBkg["All Particles Signal"] = histSignal["all"]
        multi_hist_plot(histBkg, imageOutputPath + "momSignalBkgDensityZoomedMC" + str(numFiles) + ".png", "Momentum of BKG and Signal MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (Gev)", yLabel="Percent", label="All Particles Signal", binLow=0.00001, binHigh=100, binSteps=0.3, binType="exp", barType="step",logY=True, logX=True, density=True)
        
    if args == "ptmomentumOverlay-all" or args == "":
        histSignal = momPerMC(dic, "all", "pt")
        histBkg = momPerMC(dicbkg, "all", "pt")
        histBkg["All Particles BKG"] = histBkg.pop("all")
        histBkg["All Particles Signal"] = histSignal["all"]
        multi_hist_plot(histBkg, imageOutputPath + "ptmomSignalBkgDensityZoomedMC" + str(numFiles) + ".png", "pT of BKG and Signal MC particles (" + str(numFiles) + " Files)", xLabel="pT (Gev)", yLabel="Percent", label="All Particles Signal", binLow=0.00001, binHigh=100, binSteps=0.3, binType="exp", barType="step",logY=True, logX=True, density=True)

def plotOccupancy(dic, dicbkg, args=""):
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
    
    # print(imageOutputPath)
    batch = ""
    if typeFile == "Bkg":
        batch = "20 BKG File Batch"
    elif typeFile == "Signal":
        batch = "1 Signal File Batch"
        
    if args == "occupancy-BatchedBatch" or args == "":
        hist = occupancy(dic, "occupancy_per_batch_sum_batches")
        #save numpy of occupancy per batch sum batches:
        # np.save("occupancy_per_batch_sum_batches_nonmean.npy", hist["occupancy_per_batch_sum_batches_non_meaned"])
        np.savetxt("occupancy_per_batch_sum_batches_nonmean.csv",  hist["occupancy_per_batch_sum_batches_non_meaned"], delimiter=",", fmt="%.6f")  # Adjust precision as needed
        layers = [i for i in range(0, hist["total_number_of_layers"])]
        xy_plot(layers, hist["occupancy_per_batch_sum_batches"], imageOutputPath + "occupancy"+str(typeFile)+"FileBatchMC" + str(numFiles) + ".png",
                "Average Occupancy Across Each " + batch + " (" + str(numFiles) + " Files)",
                xLabel="Radial Layer Index", yLabel="Average Channel Occupancy [%]",
                includeLegend=False, label="", scatter=True, errorBars=True, yerr = hist["occupancy_per_batch_sum_batches_error"])
    
    if args == "occupancy-BatchedBatchNN" or args == "":
        hist = occupancy(dic, "occupancy_per_batch_sum_batches_non_normalized")
        layers = [i for i in range(0, hist["total_number_of_layers"])]
        xy_plot(layers, hist["occupancy_per_batch_file_non_normalized"], imageOutputPath + "occupancy"+str(typeFile)+"FileBatchNNMC" + str(numFiles) + ".png", 
                "Average Occupancy Across " + batch + " Non Normalized (" + str(numFiles) + " Files)",
                xLabel="Radial Layer Index", yLabel="Average Channel Occupancy",
                includeLegend=False, label="", scatter=True, errorBars=True, yerr = hist["occupancy_per_batch_file_non_normalized_error"])
    
    if args == "occupancy-nCellsPerLayer" or args == "":
        hist = occupancy(dic, "cells_per_layer")
        x = [int(i) for i in list(hist["n_cells_per_layer"].keys())]
        print(f"n_cells_per_layer: {hist['n_cells_per_layer'].values()}")
        xy_plot(x, list(hist["n_cells_per_layer"].values()), imageOutputPath + "nCellsPerLayer"+str(typeFile)+"MC" + str(numFiles) + ".png", "Cells Per Layer (" + str(numFiles) + " Files)", xLabel="Layer Number", yLabel="Cells Per Layer", includeLegend=False, label="")
    
    if args == "occupancy-onlyNeighbors" or args == "":
        hist = occupancy(dic, "occupancy_per_batch_sum_batches_only_neighbor")
        layers = [i for i in range(0, hist["total_number_of_layers"])]
        xy_plot(layers, hist["occupancy_per_batch_sum_batches_only_neighbor"], imageOutputPath + "occupancy"+str(typeFile)+"FileBatchMC" + str(numFiles) + "OnlyNeighbors.png",
                "Average Occupancy Across Each " + batch + " (" + str(numFiles) + " Files)",
                xLabel="Radial Layer Index", yLabel="Average Channel Occupancy [%]",
                includeLegend=False, label="", scatter=True, errorBars=True, yerr = hist["occupancy_per_batch_sum_batches_only_neighbor_error"])
        
    if args == "avg-energy-deposit" or args == "":
        hist = occupancy(dic, "avg_energy_deposit")
        layers = [i for i in range(0, hist["total_number_of_layers"])]
        xy_plot(layers, hist["occupancy_per_batch_sum_batch_avg_energy_dep"], imageOutputPath + "energyDeposit"+str(typeFile)+"MC" + str(numFiles) + ".png",
                "Average Energy Deposit Across \nEach " + batch + " (" + str(numFiles) + " Files)",
                xLabel="Radial Layer Index", yLabel="Average Energy Deposit [GeV]",
                includeLegend=False, label="", scatter=True, errorBars=True, yerr = hist["occupancy_per_batch_sum_batch_avg_energy_dep_error"])
        
    if args == "energy-deposit" or args == "":
        hist = occupancy(dic, "energy_deposit")
        heatmap(hist["occupancy_per_batch_sum_batches_energy_dep"], imageOutputPath + "energyDepositLogHeatmap"+str(typeFile)+"MC" + str(numFiles) + ".png", 
                "Energy Deposit Across \nEach " + batch + " (" + str(numFiles) + " Files)", 
                xLabel="Radial Layer Index", yLabel="Cell Layer Index", label="Energy Deposit (Gev)", logScale=True)
                
        
        
def plotWireChamber(dic, dicbkg, args=""):
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
    
def plotHitRadius(dic, dicbkg, args=""):
    """
    Plot the hit radius of all particles.

    Args:
        dic (dictionary): _description_
        dicbkg (dictionary): _description_
        args (str, optional): _description_. Defaults to "".
    """
    if args == "hitRadius-all" or args == "":
        hist = hitRadius(dic, "All")
        # print(np.array(hist["All"]))
        print(f"max hit radius: {max(hist['All'])}")
        print(f"min hit radius: {min(hist['All'])}")
        hist_plot(hist["All"], 
                  imageOutputPath + "hitRadius" + str(typeFile) + "MC" + str(numFiles) + ".png", 
                  "Hit Radius of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", 
                  xLabel="Hit Radius (mm)", yLabel="Count MC particles", binLow=min(hist['All']), binHigh=max(hist['All']), binType="lin", binSteps=10)
    
    if args == "hitRadius-layers-radius" or args == "":
        hist = hitRadius(dic, "OverlayLayer")
        #superlayers 14
        #layers 112
        fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
        ax = fig.add_subplot(111)
        layers = np.arange(1, 113)
        radii = np.linspace(350, 2000, 112)
        # print(hist["OverlayLayer"].values())
        
        numCells = list(hist["OverlayLayer"].values())
        scatter = ax.scatter(radii, layers, c=numCells, cmap='viridis', s=20, edgecolors="black")
        # Labels
        ax.set_xlabel('Radius (mm)')
        ax.set_ylabel('Layer Number')
        # ax.set_zlabel('Number of Cells')

        # Adding color bar
        fig.colorbar(scatter, ax=ax, label="Number of Cells")

        plt.title("3D Distribution of Cells per Layer")
        # plt.show()
        plt.savefig(imageOutputPath + "hitRadiusLayersDistribution" + str(typeFile) + "MC" + str(numFiles) + ".png")
        
    if args == "hitRadius-all-layers" or args == "":
        hist = hitRadius(dic, "All")
        # print(np.array(hist["All"]))
        # print(f"max hit radius: {max(hist['All'])}")
        # print(f"min hit radius: {min(hist['All'])}")
        # hist["layers"] = [i for i in range(1, 113)]
        

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        hist_plot(hist["All"], 
                  imageOutputPath + "hitRadius" + str(typeFile) + "MC" + str(numFiles) + ".png", 
                  "Hit Radius of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", 
                  xLabel="Hit Radius (mm)", yLabel="Count MC particles", binLow=min(hist['All']), binHigh=max(hist['All']), binType="lin", binSteps=5,
                  save=False, figure=fig, axe=ax)
        # colors = ["red", "orange", "black", "yellow", "pink", "brown", "cyan", "magenta", "grey", "lime", "teal", "indigo"]
        save=False
        superLayerRadii = np.linspace(350, 2000, 15)
        superLayerHeight = [10000 for i in range(15)]
        for i in range(0, 15):
            xy_plot([superLayerRadii[i], superLayerRadii[i]], [0, superLayerHeight[i]], 
                    imageOutputPath + "hitRadiusSuperLayers" + str(typeFile) + "MC" + str(numFiles) + ".png", 
                    "Hit Radius of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", 
                    xLabel="Radius (mm)", yLabel="Count MC particles", includeLegend=False, label="", scatter=False, figure=fig, axe=ax, save=save, color="red")
            #plot the lyaers in the superlayer
            currentSuperLayerLow = (2000-350)/15 * i + 350
            currentSuperLayerHigh = (2000-350)/15 * (i+1) + 350
            layerRadii = np.linspace(currentSuperLayerLow, currentSuperLayerHigh, 9)
            layerHeight = [7500 for i in range(9)]
            for j in range(0,9):
                if i == 14 and j == 8:
                    save=True
                xy_plot([layerRadii[j], layerRadii[j]], [0, layerHeight[j]], 
                        imageOutputPath + "hitRadiusLayers" + str(typeFile) + "MC" + str(numFiles) + ".png", 
                        "Hit Radius of " + str(typeFile) + " MC particles (" + str(numFiles) + " Files)", 
                        xLabel="Radius (mm)", yLabel="Count MC particles", includeLegend=False, label="", scatter=False, figure=fig, axe=ax, save=save, color="yellow")
    
    
def plotAll(dic, dicbkg, args=""):
    """
    Plot all histograms.
    """
    ###momentum plots
    plotMomentum(dic)
    
    ###pdg plots
    plotPDG(dic)
    
    ###hits plots
    plotHits(dic)
    
    ###overlay bkg and signal mom
    plotMomOverlay(dic, dicbkg)
    
    ###overlay bkg and signal hits
    plotHitsOverlay(dic, dicbkg)
    
    ###occupancy
    plotOccupancy(dic)
    
    ###wireChamber
    plotWireChamber(dic)
    
    ###hitRadius
    plotHitRadius(dic)
    
    ###3d position
    plot3dPosition(dic)
    
    print("All plots saved")
    
def genPlot(inputArgs):
    """
    Used by the argparse to generate the desired plot(s). \n
    Basically just want to map the inputArgs to the correct function. \n
    For a given input, it will call the corresponding plot-function which will delegate to the correct part of function and plot. \n
    Note though I tried to be as general with plotting, there are some instances where I use specific values which may need to be changed depending on the data. 
    
    
    Inputs: inputArgs, should be one argument either "" or from typePlots
            imageOutputPath, path to save the image
    Outputs: plot saved to imageOutputPath
    """
    if len(inputArgs) != 2 and len(inputArgs) != 3 and len(inputArgs) != 4:
        raise argparse.ArgumentTypeError("The --plot argument requires at least 2 arguments, the type of plot to generate, the type of files to use (optional 3rd argument includeBkg).")
    
    typePlot = inputArgs[0]
    typefile = inputArgs[1]
    if len(inputArgs) > 2 and inputArgs[2].isdigit():
        numFiles=inputArgs[2]
    includeBkg = bool(inputArgs[3]) if len(inputArgs) == 4 else False
    dic, dicbkg = setup(typefile, includeBkg, numFiles)
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
        "ptmomentum-all": plotMomentum,
        "ptmomentum-onlyOH": plotMomentum,
        "ptmomentum-only+H": plotMomentum,
        "ptmomentum-onlyParPhoton": plotMomentum,
        "ptmomentum-ptBelow10R": plotMomentum,
        "ptmomentum-onlyOH-pdg": plotMomentum,
        "ptmomentum-only+H-pdg": plotMomentum,
        "ptmomentum-multiHits": plotMomentum,
        "ptmomentum-multiHitsExcludeOne": plotMomentum,
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
        "ptmomentumOverlay-all": plotMomOverlay,
        "occupancy-nCellsFired": plotOccupancy,
        "occupancy-BatchedBatch": plotOccupancy,
        "occupancy-BatchedBatchNN": plotOccupancy,
        "occupancy-nCellsPerLayer": plotOccupancy,
        "occupancy-onlyNeighbors": plotOccupancy,
        "avg-energy-deposit": plotOccupancy,
        "energy-deposit": plotOccupancy,
        "wireChamber-all": plotWireChamber,
        "plot3dPos": plot3dPosition,
        "hitRadius-all": plotHitRadius,
        "hitRadius-layers-radius": plotHitRadius,
        "hitRadius-all-layers": plotHitRadius
    }

    # Execute the corresponding function if the key exists
    if typePlot in function_map:
        function_map[typePlot](dic, dicbkg, typePlot)  # Calls func_b()
        print(f"Saving to: {imageOutputPath}")
    else:
        print("No match found")
          
       
#'''
#create argument parser so someone can create plots without hard coding
parser = argparse.ArgumentParser()
typePlots = ["", "all", 
             "momentum-all", "momentum-onlyOH", "momentum-only+H",
                "momentum-onlyParPhoton", "momentum-ptBelow10R", "momentum-onlyOH-pdg",
                "momentum-only+H-pdg", "momentum-multiHits", "momentum-multiHitsExcludeOne",
                "pdg-all", "pdg-electron", "pdg-gen",
                "hits-all", "hits-neutron", "hits-photon", "hits-electron",
                "hitsOverlay-all", "hitsOverlay-electron", "hitsOverlay-photon",
                "momentumOverlay-all",
                "occupancy-nCellsFired", "occupancy-BatchedBatch", "occupancy-BatchedBatchNN",
                "occupancy-nCellsPerLayer", "occupancy-onlyNeighbors", "avg-energy-deposit",
                "energy-deposit",
                "wireChamber-all",
                "plot3dPos", "hitRadius-all", "hitRadius-layers-radius", "hitRadius-all-layers"
             ]
parser.add_argument('--plot', help="Inputs... \n-- plotType(str): " +
                    str(typePlots) + 
                    "\n-- fileType(str): [Bkg], [Signal], [Combined]" + 
                    "\n-- numFiles(int): Default(500)" +
                    "\n-- includeBkg(optional: Bool): [True] [False]", type=str, default="", nargs='+')
args = parser.parse_args()

if args.plot and args.plot != "":
    try:
        print(f"Parsed --plot arguments: {args.plot}")
        genPlot(args.plot)
    except ValueError as e:
        parser.error(str(e))
#'''
'''    
example (plot all):
/work/submit/poulin/fccproject-tracking/detector_beam_backgrounds/tracking/trBkgPlt.py --plot all Bkg

example(plot momentum)
/work/submit/poulin/fccproject-tracking/detector_beam_backgrounds/tracking/trBkgPlt.py --plot momentum-all Signal
'''