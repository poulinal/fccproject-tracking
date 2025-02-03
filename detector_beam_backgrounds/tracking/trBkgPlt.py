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


available_functions = ["hitsPerMC", "momPerMC", "PDGPerMC", "wiresPerMC", "trajLen", "radiusPerMC", "angleHits", "occupancy"]
dic = {}
dicbkg = {}
numFiles = 500
# backgroundDataPath = "fccproject-tracking/detector_beam_backgrounds/tracking/data/bkg_background_particles_"+str(numFiles)+".npy"
backgroundDataPath = "fccproject-tracking/detector_beam_backgrounds/tracking/data/occupancy_tinker/bkg_background_particles_"+str(numFiles)+".npy"
combinedDataPath = "fccproject-tracking/detector_beam_backgrounds/tracking/data/combined/"
imageOutputPath = "fccproject-tracking/detector_beam_backgrounds/tracking/images/test"
signalDataPath = "fccproject-tracking/detector_beam_backgrounds/tracking/data/occupancy_tinker/signal_background_particles_"+str(numFiles)+".npy"

#change to personal directories in here:
def setup(type: str ="bkg", includeBkg: bool =False):
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
    
    Keyword arguments:
    type -- what type of data to load, either background, combined, or signal \n
    includeBkg -- for when we want to overlay bkg and signal files in a plot \n
    Return: no return, just updates the global dictionary dic
    """
    if type == "bkg":
        dic = np.load(backgroundDataPath, allow_pickle=True).item()
    elif type == "combined":
        dic = np.load(combinedDataPath, allow_pickle=True).item()
    elif type == "signal":
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

def momPerMC(dic, args = "", byPDG = False):
    """
    Given a hit by a particle, what is that particles momentum.
    Inputs: dic from setup,
            args is the type of hits to calculate:
                "" -- calculate all args \n
                "all" -- all particles \n
                "onlyOH" -- all particles with only one hit \n
                "only+H" -- all particles with more than one hit \n
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
                
def PDGPerMC(dic, args = "", sepSecondary = False):
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

def wiresPerMC(dic):
    """
    Function that returns the number of wires per MC particle.
    WIP
    Inputs: dic, 
    Outputs: hist, histogram with the number of wires per MC particle.
    """
    hist = []
    return hist

def trajLen(dic):
    """
    Function that returns the trajectory length of MC particles.
    WIP
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the trajectory length of MC particles.
    """
    hist = []
    #get the length of the vertex to the first hit
    # mcParticlesPosHit = groupHits(dic, dic["pos_z"])
    mcParticlesPosHit  = []
    totMCPL = []
    #go through each mcParticle, 
    for i in range(0, len(mcParticlesPosHit)):
        # given mcParticlesPosHit is an array (each particle) of 3position arrays (each hit)
        # get just the max z and min z and subtract them
        # print(f"mcPPH[i]: {mcParticlesPosHit[i][0]}")
        totPL = abs(np.max(mcParticlesPosHit[i]) - np.min(mcParticlesPosHit[i]))
        
            
        totMCPL.append(totPL)
    # print(f"totMCPL: {totMCPL}")
    # print(f"max: {max(totMCPL)}")
    #hist = ROOT.TH1F("hist", "Trajectory length of MC particles", 40, 0, max(totMCPL))
    
    #make numpy array:
    totMCPL = np.array(totMCPL)
    #remove zeros
    totMCPL = totMCPL[totMCPL != 0]
    # print(f"totMCPL after rm 0: {totMCPL}")
    
    for i in range(0, len(totMCPL)):
        hist.append(totMCPL[i])
    return hist, totMCPL

def radiusPerMC(dic):
    """
    Function that returns the radius of MC particles.
    WIP
    Inputs: dic, 
    Outputs: hist, histogram with the radius of MC particles.
    """
    hist = []
    for i in range(0, len(dic["R"])):
        hist.append(dic["R"][i])
    return hist

def angleHits(dic):
    """
    Function that returns the angle of hits.
    WIP
    Inputs: dic,
    Outputs: hist, histogram with the angle of hits.
    """
    hist = []
    for i in range(0, len(dic["px"])):
        hist.append(phi(dic["px"][i], dic["py"][i]))
    return hist

def phi(pos_vec):
    """
    Function that calculates the angle of a vector.
    Inputs: pos_vec, 3D vector.
    Outputs: phi, angle of the vector.
    """
    phi = math.atan(pos_vec[1]/pos_vec[0])
    if pos_vec[0] < 0:
        phi += math.pi
    elif pos_vec[1] < 0:
        phi += 2*math.pi
    return phi
    
    
    

# hist = momPerMC(dic, "")
# hist_plot(hist['All'], imageOutputPath + "momentumMC" + str(numFiles) + ".png", "Momentum of MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles")
# hist_plot(hist['All'], imageOutputPath + "momentumMC" + str(numFiles) + "Loggedx.png", "Momentum of MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# hist_plot(hist['All'], imageOutputPath + "momentumMC" + str(numFiles) + "Logged.png", "Momentum of MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True, logY=True)

# hist = momPerMC(dic, "onlyOH")
# hist_plot(hist["onlyOH"], imageOutputPath + "momentumMC" + str(numFiles) + "onlyOneHit.png", "Momentum of MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles")
# hist_plot(hist["onlyOH"], imageOutputPath + "momentumMC" + str(numFiles) + "onlyOneHitLoggedx.png", "Momentum of MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# hist_plot(hist["onlyOH"], imageOutputPath + "momentumMC" + str(numFiles) + "onlyOneHitLogged.png", "Momentum of MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)

# hist = momPerMC(dic, "only+H")
# hist_plot(hist["only+H"], imageOutputPath + "momentumMC" + str(numFiles) + "only+Hit.png", "Momentum of MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles",  xMin=0, xMax=0.3)
# hist_plot(hist["only+H"], imageOutputPath + "momentumMC" + str(numFiles) + "only+HitLoggedx.png", "Momentum of BKG MC particles \nwith more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# hist_plot(hist["only+H"], imageOutputPath + "momentumMC" + str(numFiles) + "only+HitLogged.png", "Momentum of MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)

# hist = momPerMC(dic, "onlyParPhoton")
# hist_plot(hist, imageOutputPath + "momentumMC" + str(numFiles) + "onlyParentPhoton.png", "Momentum of MC particles with a Parent Photon (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles")
# hist_plot(hist, imageOutputPath + "momentumMC" + str(numFiles) + "onlyParentPhotonLoggedx.png", "Momentum of MC particles with a Parent Photon (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# hist_plot(hist, imageOutputPath + "momentumMC" + str(numFiles) + "onlyParentPhotonLogged.png", "Momentum of MC particles with a Parent Photon (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)

# hist = momPerMC(dic, "ptBelow10R")
# hist_plot(hist, imageOutputPath + "momentumMC" + str(numFiles) + "pt10R.png", "Transverse Momentum of MC particles within a radius of 10mm (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles")
# hist_plot(hist, imageOutputPath + "momentumMC" + str(numFiles) + "pt10RLoggedx.png", "Transverse Momentum of MC particles within a radius of 10mm (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# hist_plot(hist, imageOutputPath + "momentumMC" + str(numFiles) + "pt10RLogged.png", "Transverse Momentum of MC particles within a radius of 10mm (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)




####multi:
hist = PDGPerMC(dic, "")
bar_plot(hist.keys(), hist.values(), imageOutputPath + "pdgMC" + str(numFiles) + ".png", "PDG of BKG MC particles \n(" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", width=1.3, rotation=90, includeLegend=False)

# hist = PDGPerMC(dic, "electron")
# bar_plot(["electron"], hist["all"], imageOutputPath + "pdgElectronPhotonMC" + str(numFiles) + ".png", "PDG of MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80, save=False, label="All Electrons")
# bar_plot(["electron"], hist["electron"], imageOutputPath + "pdgElectronPhotonMC" + str(numFiles) + ".png", "PDG of MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80, save=False, label="All Electrons")
# bar_plot(["electron"], hist["e_photon_parent"], imageOutputPath + "pdgElectronPhotonMC" + str(numFiles) + ".png", "PDG of MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80, label="Only Electrons with a Parent Photon")

# hist = PDGPerMC(dic, "gen")
# bar_plot("MC Particles", hist["all"], imageOutputPath + "pdgGeneratorStatusMC" + str(numFiles) + ".png", "Primary or Secondary MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80, save=False, label="All Particles")
# multi_bar_plot("MC Particles", hist, imageOutputPath + "pdgElectronGeneratorStatusMC" + str(numFiles) + ".png", "Primary or Secondary for Electron MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles")

######3maybe return entire dictionary and in plotting remove zero valued keys
# hist = hitsPerMC(dic, "neutronSec")
# print(f"hist: {max(hist['all'])}")
# hist_plot(hist["all"], imageOutputPath + "hitsMC" + str(numFiles) + ".png", "Hits of MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Number of MC particles", label="All Particles", autoBin=False, logX=True, logY=True)
# multi_hist_plot(hist, imageOutputPath + "hitsNeutronMC" + str(numFiles) + ".png", "Hits of MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Number of MC particles", label="All Particles", barType="step", autoBin=False, binLow=0.1, binHigh=9000, binSteps=5, binType="lin", logY=True, contrast=True)
# multi_hist_plot(hist, imageOutputPath + "hitsNeutronZoomedMC" + str(numFiles) + ".png", "Hits of MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Number of MC particles", label="All Particles", barType="step", autoBin=False, binLow=0.1, binHigh=900, binSteps=5, binType="lin", logY=True)
# multi_bar_plot("MCP", hist, imageOutputPath + "hitsPhotonMC" + str(numFiles) + ".png", "Hits of MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80,  
#          additionalText=f"Percentage of photons that were produced by secondary particles: {round(hist['Only Photons Produced by Secondary Particles']/hist['Only Photons'], 3)}")




####byPDG
# hist = momPerMC(dic, "onlyOH", byPDG=True)
# print(max(hist["onlyOH"]))
# multi_hist_plot(hist, imageOutputPath + "momentumMC" + str(numFiles) + "onlyOneHitSepPDG.png", "Momentum of MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", barType="step")
# multi_hist_plot(hist, imageOutputPath + "momentumMC" + str(numFiles) + "onlyOneHitSepPDGLoggedx.png", "Momentum of MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True, barType="step", contrast=True)
# multi_hist_plot(hist, imageOutputPath + "momentumMC" + str(numFiles) + "onlyOneHitSepPDGLogged.png", "Momentum of MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True, barType="step")

# hist = momPerMC(dic, "only+H", byPDG=True)
# print(f"hist: {hist['only+H']}")
# multi_hist_plot(hist, imageOutputPath + "momentumMC" + str(numFiles) + "only+Hit.png", "Momentum of BKG MC particles \n with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles",  xMin=0, xMax=0.3, barType="step")
# multi_hist_plot(hist, imageOutputPath + "momentum" + str(type) + "MC" + str(numFiles) + "only+HitSetpPDGLoggedx.png", "Momentum of " + str(type) + " MC particles \nwith more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True, barType="step")
# multi_hist_plot(hist, imageOutputPath + "momentumMC" + str(numFiles) + "only+HitLogged.png", "Momentum of MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True, barType="step")

# hist = momPerMC(dic, "multiHitsExcludeOne")
# hist = momPerMC(dic, "multiHits")
# multi_hist_plot(hist, imageOutputPath + "momentumMC" + str(numFiles) + "MultiHitsExcludeOneLoggedx.png", "Momentum of MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# multi_hist_plot(hist, imageOutputPath + "momentumMC" + str(numFiles) + "MultiHitsLoggedx.png", "Momentum of MC particles with hits (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)



###overlay bkg and signal hits
# histSignal = hitsPerMC(dic, "electron")
# histBkg = hitsPerMC(dicbkg, "electron")
# histBkg["Electron Particles Signal"] = histSignal["Only Electrons"]
# histBkg["Electron Particles BKG"] = histBkg.pop("Only Electrons")
# histSignal["Electron Particles Signal"] = histSignal.pop("Only Electrons")
# bar_step_multi_hist_plot(histSignal["all"], histBkg, imageOutputPath + "hitsSignalBkgMC" + str(numFiles) + ".png", "Hits of MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Number of MC particles", label="All Particles Signal", autoBin=False, binLow=0.1, binHigh=9000, binSteps=5, binType="lin", logY=True)
# histBkg.pop("all")
# multi_hist_plot(histBkg, imageOutputPath + "hitsElectronDensitySignalBkgZoomedMC" + str(numFiles) + ".png", "Hits of MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Density of MC particles", label="All Particles Signal", autoBin=False, binLow=0.1, binHigh=900, binSteps=5, binType="lin", barType="step",logY=True, density=True)
# multi_hist_plot(histBkg, imageOutputPath + "hitsDensitySignalBkgZoomedMC" + str(numFiles) + ".png", "Hits of MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Density of MC particles", label="All Particles Signal", autoBin=False, binLow=0.1, binHigh=900, binSteps=5, binType="lin", barType="step", logY=True, density=True)

###overlay bkg and signal mom
# histSignal = momPerMC(dic, "All")
# histBkg = momPerMC(dicbkg, "All")
# histBkg["All Particles BKG"] = histBkg.pop("All")
# histBkg["All Particles Signal"] = histSignal["All"]
# print(f"max histSignal: {max(histBkg['All Particles Signal'])}")
# multi_hist_plot(histBkg, imageOutputPath + "momSignalBkgDensityZoomedMC" + str(numFiles) + ".png", "Momentum of MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (Gev)", yLabel="Density of MC particles", label="All Particles Signal", autoBin=False, binLow=0.0001, binHigh=50, binSteps=0.3, binType="exp", barType="step",logY=True, logX=True, density=True)


#####occupancy
# hist = occupancy(dic, "avg_occupancy_layer_file")
# hist = occupancy(dic, "n_cells")
#total number of layers: 112
#total number of cells 56448

# print(f"hist['percentage_fired']: {hist['percentage_fired']}")
# print(f"max(n_cells): {hist['n_cells']}")
# hist_plot(hist["n_cells"], imageOutputPath + "nCellsFiredMC" + str(numFiles) + ".png", "Number of Cells Fired by Particles (" + str(numFiles) + " Files)", xLabel="Number of cells fired by an MC particle", yLabel="Count", xMin=0, xMax=4000, binLow=0.01, binHigh=4000, binSteps=0.3, binType="lin")
# hist_plot(hist["percentage_fired"], imageOutputPath + "occupancyPercMC" + str(numFiles) + ".png", "Occupancy of the detector (" + str(numFiles) + " Files)", xLabel="Percentage of cells fired", yLabel="Count MC particles", xMin=0, xMax=80, binHigh=80, binSteps=1, binType="lin")
# hist_plot(hist["avg_occupancy_file"], imageOutputPath + "occupancyPerFileMC" + str(numFiles) + ".png", "Average Occupancy Per File (" + str(numFiles) + " Files)", xLabel="Radial Layer Index", yLabel="Average Channel Occupancy [%]", xMin=0, xMax=112, binHigh=112, binSteps=1, binType="lin")
# hist_plot(hist["avg_occupancy_event"], imageOutputPath + "occupancyPerEventMC" + str(numFiles) + ".png", "Average Occupancy Per Event (" + str(numFiles) + " Files)", xLabel="Radial Layer Index", yLabel="Average Channel Occupancy [%]", xMin=0, xMax=1.3, binHigh=112, binSteps=1, binType="lin")

# print(hist)
# hist = occupancy(dic, "occupancy_per_batch_sum_batches")
# layers = [i for i in range(0, hist["total_number_of_layers"])]

# print(hist["occupancy_per_batch_file_non_normalized"])
# xy_plot(layers, hist["occupancy_per_batch_file_non_normalized"], imageOutputPath + "occupancy20FileBatchNNMC" + str(numFiles) + ".png", 
#         "Average Occupancy Across 20 File Non Normalized (" + str(numFiles) + " Files)",
#         xLabel="Radial Layer Index", yLabel="Average Channel Occupancy", 
#         includeLegend=False, label="", scatter=True, errorBars=True, yerr = hist["occupancy_per_batch_file_non_normalized_error"])
# xy_plot(layers, hist["occupancy_per_batch_sum_batches"], imageOutputPath + "occupancy20FileBatchMC" + str(numFiles) + ".png", 
#         "Average Occupancy Across \nEach 20 BKG File Batch (" + str(numFiles) + " Files)",
#         xLabel="Radial Layer Index", yLabel="Average Channel Occupancy [%]", 
#         includeLegend=False, label="", scatter=True, errorBars=True, yerr = hist["occupancy_per_batch_sum_batches_error"])


# x = [int(i) for i in list(hist["n_cells_per_layer"].keys())]
# xy_plot(x, list(hist["n_cells_per_layer"].values()), imageOutputPath + "nCellsPerLayerMC" + str(numFiles) + ".png", "Cells Per Layer (" + str(numFiles) + " Files)", xLabel="Layer Number", yLabel="Cells Per Layer", includeLegend=False, label="")

###plot wire chamber
# hist = occupancy(dic, "cells_per_layer")
# print(hist['total_number_of_layers'])
# print(len(hist['n_cells_per_layer']))
# plot_wire_chamber(hist["total_number_of_layers"], hist["n_cells_per_layer"], imageOutputPath + "wireChamberFirstQuad" + ".png", title="", firstQuadrant=True)




    
    
def genPlot(inputArgs, imageOutputPath):
    """
    Used by the argparse to generate the desired plot(s).
    Inputs: inputArgs, should be one argument either "" or from typePlots
            imageOutputPath, path to save the image
    Outputs: plot saved to imageOutputPath
    """
    if len(inputArgs) < 2:
        raise argparse.ArgumentTypeError("The --plot argument requires at least hist.")
    # Convert the three inputs to the appropriate types
    #match argument to correct variable based on type
    
    hist = inputArgs[0]  # String
    #check if hist in available_functions
    if hist not in available_functions:
        raise argparse.ArgumentTypeError(f"The {hist} is not a member of available functions.")
    
    outname = inputArgs[1] # String
    title = inputArgs[2]

    #check if xmin, xmax are empty
    if inputArgs[3] == "":
        emX = True
    else:
        xMin = float(inputArgs[3])
        xMax = float(inputArgs[4])
    if inputArgs[5] == "":
        emY = True
    else:
        yMin = float(inputArgs[5])
        yMax = float(inputArgs[6])
        
    xLabel = inputArgs[7]  # String
    yLabel = inputArgs[8]  # String 
    
    #check if ylog is empty
    if inputArgs[9] == "":
        emYlog = True
    else:
        ylog = bool(inputArgs[9])  # Boolean
    
    if emX:
        if emY:
            plot_hist(hist, imageOutputPath + "" + outname + ".png", title, xLabel, yLabel, ylog)
    if emY:
        plot_hist(hist, imageOutputPath + "" + outname + ".png", title, xMin, xMax, yLabel, ylog)
    else:
        plot_hist(hist, imageOutputPath + "" + outname + ".png", title, xMin, xMax, yMin, yMax, xLabel, yLabel, ylog)
          
       
#'''
parser = argparse.ArgumentParser()
typePlots = ["all", "momentum-all", "momentum-onlyOH", "momentum-only+H", "momentum-onlyParPhoton", "momentum-ptBelow10R", "PDGPerMC", "hitsPerMC", "occupancy"]
parser.add_argument('--plot', help="Plot histogram -- type(str): " +
                    str(typePlots), type=str, default="", nargs='1')
args = parser.parse_args()

outputFolder = imageOutputPath + ""

if args.plot and args.plot != "":
    try:
        print(f"Parsed --plot arguments: ...")
        genPlot(args.plot, outputFolder)
    except ValueError as e:
        parser.error(str(e))
#'''
'''    
   
/work/submit/poulin/fccproject-tracking/detector_beam_backgrounds/tracking/trBkgPlt.py --plot

#plot hits per mc
hist hitsMC Number-of-hits-per-MC-particle Number-of-hits Number-of-MC-particles "" "" "" "" Number-of-hits Number-of-MC-particles False


'''