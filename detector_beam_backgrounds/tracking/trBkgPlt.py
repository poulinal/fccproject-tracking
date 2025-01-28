import ROOT
import numpy as np 
from utilities.functions import hist_plot, plot_hist, multi_hist_plot, bar_plot, multi_bar_plot, xy_plot
import argparse
import sys
import matplotlib.pyplot as plt
import math


available_functions = ["hitsPerMC", "momPerMC", "pathLenWireMC", "totPathLenMC", "wiresPerMC", "trajLen", "radiusPerMC", "angleHits"]

numFiles = 20
backgroundDataPath = "fccproject-tracking/detector_beam_backgrounds/tracking/data/bkg_background_particles_"+str(numFiles)+".npy"
combinedDataPath = "fccproject-tracking/detector_beam_backgrounds/tracking/data/combined/"
imageOutputPath = "fccproject-tracking/detector_beam_backgrounds/tracking/images/"

dic = np.load(backgroundDataPath, allow_pickle=True).item()
#dic = np.load("fccproject-tracking/detector_beam_backgrounds/tracking/data/background_particles_20.npy", allow_pickle=True).item()


def hitsPerMC(dic, args = ""):
    """
    Function that returns the number of hits per MC particle.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, dictionary of values (count of hits) with keys: \n
        "all" -- all particles \n
        "photon" -- all particles with pdg 22 \n
        "photonSec" -- all particles with pdg 22 and produced secondary particles \n
        "neutron" -- all particles with pdg 2112 \n
        "neutronSec" -- all particles with pdg 2112 and produced secondary particles \n
        "electron" -- all particles with pdg 11 \n
        "allPDG" -- all particles with pdg as key and count as value \n
        "multiHits" -- all particles with keys of one hit, >1 hit, >5 hits, >10 hits, >20 hits \n
    """
    hist = {}
    hist["all"] = []
    #set mcParticles to be numpy array size of the last entry of dic["hits"]
    list_hits_per_mc = dic["count_hits"]
    print(f"max list_hits_per_mc: {max(list_hits_per_mc)}")

    # print(f"len list_hits_per_mc: {len(list_hits_per_mc)}")
    
    # for i in range(0, len(list_hits_per_mc)):
    #     hist["all"] += 1
    hist["all"] = list_hits_per_mc
    
    pdg = dic["pdg"]
    
    if args.startswith("photon"): #return only the hits that have a pdg photon
        hist["Only Photons"] = []
        for i in range(0, len(list_hits_per_mc)):
            if pdg[i] == 22:
                hist["Only Photons"].append(list_hits_per_mc[i])
        
        
        if args == "photonSec":
            hist["Only Photons Produced by Secondary Particles"] = []
            hits_produced_secondary = dic["hits_produced_secondary"]
            hits_mc_produced_secondary = dic["hits_mc_produced_secondary"]
            
            # print(f"len pdg: {len(pdg)}")
            # print(f"len produced_secondary: {len(hits_produced_secondary)}") 
            # print(f"len hits_mc_produced_secondary: {len(hits_mc_produced_secondary)}")
            
            for i in range(0, len(list_hits_per_mc)):
                if pdg[i] == 22 and hits_mc_produced_secondary[i]:
                    hist["Only Photons Produced by Secondary Particles"].append(list_hits_per_mc[i])
            #print(f"hits: {hits_mc_produced_secondary}")
            
    if args.startswith("neutron"):
        hist["Only Neutrons"] = []
        for i in range(0, len(list_hits_per_mc)):
            if pdg[i] == 2112:
                hist["Only Neutrons"].append(list_hits_per_mc[i])
        
        if args == "neutronSec":
            hist["Only Neutrons Produced by Secondary Particles"] = []
            hits_mc_produced_secondary = dic["hits_mc_produced_secondary"]
            
            for i in range(0, len(list_hits_per_mc)):
                if pdg[i] == 2112 and hits_mc_produced_secondary[i]:
                    hist["Only Neutrons Produced by Secondary Particles"].append(list_hits_per_mc[i])
                    
    if args == "electron":
        hist["Only Electrons"] = []
        print(f"len pdg: {len(pdg)}")
        print(f"len list_hits_per_mc: {len(list_hits_per_mc)}")
        
        for i in range(0, len(list_hits_per_mc)):
            if pdg[i] == 11:
                hist["Only Electrons"].append(list_hits_per_mc[i])
        
        hist["Only Electrons Produced by Secondary Particles"] = []
        hits_mc_produced_secondary = dic["hits_mc_produced_secondary"]
        
        for i in range(0, len(list_hits_per_mc)):
            if pdg[i] == 11 and hits_mc_produced_secondary[i]:
                hist["Only Electrons Produced by Secondary Particles"].append(list_hits_per_mc[i])
                    
    if args == "allPDG":
        pdg_unique = np.unique(pdg)
        # print(f"pdg_unique: {pdg_unique}")
        # print(f"len pdg_unique: {len(pdg_unique)}")
        # print(f"len pdg: {len(pdg)}")
        # print(f"len list_hits_per_mc: {len(list_hits_per_mc)}")
        for i in range(0, len(pdg_unique)):
            hist[pdg_unique[i]] = 0
        #print all keys in hist
        # print(f"hist keys: {hist.keys()}")
        for i in range(0, len(list_hits_per_mc)):
            #print(f"pdg[i]: {pdg[i]}")
            hist[pdg[i]] += 1
    
    if args == "multiHits":
        hist["1 Hit"] = []
        hist[">1 Hits"] = []
        hist[">5 Hits"] = []
        hist[">10 Hits"] = []
        hist[">20 Hits"] = []
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
    Function that returns the momentum of MC particles.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens. \n
    arguments can either be "", "onlyOH", "only+H", "onlyParPhoton", "ptBelow10R" \n
    Outputs: hist, dictionary of values (count of hits) with keys: \n
        "all" -- all particles momentum \n
        "onlyOH" -- all particles momentum with only one hit \n
        "only+H" -- all particles momentum with more than one hit \n
        "onlyParPhoton" -- all particles momentum with a parent photon \n
        "ptBelow10R" -- all particles momentum with a vertex radius below 10 \n
        "multiHits" -- all particles with keys of one hit, >1 hit, >5 hits, >10 hits, >20 hits \n
        "multiHitsExcludeOne" -- all particles with keys of >1 hit, >5 hits, >10 hits, >20 hits \n
    """
    # Create the histogram
    hist = {}
    hist["all"] = []
    p = dic["p"]
    
    if byPDG:
        pdg = dic["pdg"]
    
    if args == "": #regular get all momenta
        # for i in range(0, len(p)):
        #     hist.append(p[i])
        hist["all"] = p
        # print(f"max hist: {max(hist)}")
        return hist["all"]


    if args == "onlyOH" or args == "only+H":
        count_hits = dic["count_hits"] #the index of the mcParticle for each hit
        hist["onlyOH"] = []
        hist["only+H"] = []
        #seperate hits based on if they occur once or more than once with the same mcParticle
        for i in range(0, len(count_hits)):
            if count_hits[i] == 1:
                if byPDG:
                    if pdg[i] in hist:
                        hist[pdg[i]].append(p[i])
                    else:
                        hist[pdg[i]] = [p[i]]
                else:
                    hist["onlyOH"].append(p[i])
            else:
                if byPDG:
                    if pdg[i] in hist:
                        hist[pdg[i]].append(p[i])
                    else:
                        hist[pdg[i]] = [p[i]]
                hist["only+H"].append(p[i])
            hist["all"].append(p[i])
        # print(f"max hist: {max(hist)}")
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
                hist.append(p[i])
        # print(f"max hist: {max(hist)}")
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
        
        # print(f"max R: {max(R) * 0.01}")
        #print how many R's (meters) above 10 (cm)
        # print(f"R's above 10: {len(R[R > 0.01])}")
        for i in range(0, len(R)):
            if R[i] < 0.01:
                hist.append(math.sqrt(px[i]**2 + py[i]**2))
        # print(f"max hist: {max(hist)}")
        return hist
    
    if args == "multiHits" or args == "multiHitsExcludeOne":
        hist["1 Hit"] = []
        hist[">1 Hits"] = []
        hist[">5 Hits"] = []
        hist[">10 Hits"] = []
        hist[">20 Hits"] = []
        count_hits = dic["count_hits"]
        print(f"max count_hits: {max(count_hits)}")
        for i in range(0, len(count_hits)):
            print(f"count_hits[i]: {count_hits[i]}")
            if count_hits[i] == 1 and not args.endswith("ExcludeOne"):
                hist["1 Hit"].append(p[i])
            elif count_hits[i] > 1:
                hist[">1 Hits"].append(p[i])
            elif count_hits[i] > 5:
                hist[">5 Hits"].append(p[i])
            elif count_hits[i] > 10:
                hist[">10 Hits"].append(p[i])
            elif count_hits[i] > 20:
                hist[">20 Hits"].append(p[i])
        # print(f"max hist: {max(hist)}")
        return hist
        
                
def PDGPerMC(dic, args = "", sepSecondary = False):
    """
    Function that returns the PDG of MC particles.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the PDG of MC particles.
    """
    # Create the histogram
    pdg = dic["pdg"]
    prodSec = dic["hits_mc_produced_secondary"]
    hist_dic = {}
    if args == "": ##want to return all pdg's as a dictionary of pdg, key = pdg, value = count
        #hist_dic = {}
        for i in range(0, len(pdg)):
            if str(pdg[i]) in hist_dic:
                hist_dic[str(pdg[i])] += 1
            else:
                hist_dic[str(pdg[i])] = 1
        return hist_dic
    if args == "electron":
        """
        Given there was a hit, add a count if the pdg is an electron. 
        If the pdg of the particle is an electron AND has a parent photon, add a count to hist
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        hist_dic["electron"] = 0
        hist_dic["e_photon_parent"] = []
        has_par_photon = dic["has_par_photon"]
        print(len(has_par_photon))
        print(len(pdg))
        for i in range(0, len(pdg)):
            if pdg[i] == 11:
                hist_dic["electron"] += 1
                if has_par_photon[i]:
                    hist_dic["e_photon_parent"].append(pdg[i])
        return hist_dic
    if args == "gen":
        """
        given generator status, return a dictionary with the following:
        key: all, value: all particles
        key: primary, value: particles with generator status 1
        key: secondary, value: particles with generator status 2
        Keyword arguments:
        argument -- description
        Return: dictionary with keys:
        "all" -- all particles
        ...
        """
        gen = dic["gens"]
        #print full numpy
        np.set_printoptions(threshold=sys.maxsize)
        print(f"gen: {gen}")
        input("Press Enter to continue...")
        #print only non zero entries
        print(f"gen: {gen[gen != 0]}")
        input("Press Enter to continue...")
        hist_dic["all"] = 0
        hist_dic["primary"] = 0
        hist_dic["secondary"] = 0
        
        #gen 0 is created by simulation
        #gen 1 is original particles
        
        for i in range(0, len(gen)):
            hist_dic["all"] += 1
            if gen[i] == 1:
                hist_dic["primary"] += 1
            if gen[i] != 1:
                hist_dic["secondary"] += 1
        return hist_dic


def groupHits(dic, list_dic):
    """
    Function that groups the hits by MC particle.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: arrays which have the hits grouped by MC particle.
    """
    list_index = dic["hits"]
    
    list = list_dic
    mcParticlesID = []
    mcParticlesPosHit = []
    
    print(list)
    
    for i in range(0, len(list_index)):
        newEntry = []
        if list_index[i] not in mcParticlesID:
            mcParticlesID.append(list_index[i])
            newEntry = [list[i]]
            mcParticlesPosHit.append(newEntry)
        else:
            mcParticlesPosHit[mcParticlesID.index(list_index[i])].append(list[i])
    #print(f"mcParticlesPosHit: {mcParticlesPosHit}")
    return mcParticlesPosHit

def wiresPerMC(dic):
    """
    Function that returns the number of wires per MC particle.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the number of wires per MC particle.
    """
    # Create the histogram
    #hist = ROOT.TH1F("hist", "Number of wires per MC particle", 100, 0, 100)
    hist = []
    # Fill the histogram
    list_index = dic["hits"]
    mcParticles = np.zeros(max(list_index))
    for i in range(0, len(list_index)):
        mcParticles[list_index[i] - 1] += 1
    mcParticles = mcParticles[mcParticles != 0]
    for i in range(0, len(mcParticles)):
        hist.append(mcParticles[i])
    return hist

def trajLen(dic):
    """
    Function that returns the trajectory length of MC particles.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the trajectory length of MC particles.
    """
    hist = []
    #get the length of the vertex to the first hit
    mcParticlesPosHit = groupHits(dic, dic["pos_z"])
    totMCPL = []
    #go through each mcParticle, 
    for i in range(0, len(mcParticlesPosHit)):
        # given mcParticlesPosHit is an array (each particle) of 3position arrays (each hit)
        # get just the max z and min z and subtract them
        print(f"mcPPH[i]: {mcParticlesPosHit[i][0]}")
        totPL = abs(np.max(mcParticlesPosHit[i]) - np.min(mcParticlesPosHit[i]))
        
            
        totMCPL.append(totPL)
    print(f"totMCPL: {totMCPL}")
    print(f"max: {max(totMCPL)}")
    #hist = ROOT.TH1F("hist", "Trajectory length of MC particles", 40, 0, max(totMCPL))
    
    #make numpy array:
    totMCPL = np.array(totMCPL)
    #remove zeros
    totMCPL = totMCPL[totMCPL != 0]
    print(f"totMCPL after rm 0: {totMCPL}")
    
    for i in range(0, len(totMCPL)):
        hist.append(totMCPL[i])
    return hist, totMCPL

def radiusPerMC(dic):
    """
    Function that returns the radius of MC particles.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the radius of MC particles.
    """
    # Create the histogram
    hist = []
    #mcParticleR = groupHits(dic, dic["R"])
    # Fill the histogram
    #print(max(dic["R"]))
    for i in range(0, len(dic["R"])):
        hist.append(dic["R"][i])
    return hist

def angleHits(dic):
    """
    Function that returns the angle of hits.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the angle of hits.
    """
    # Create the histogram
    hist = []
    # Fill the histogram
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

def occupancy(dic, args = ""):
    """
    Function that returns the occupancy of the detector.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, dictionary of values (count of hits) with keys: \n
        "" -- all values \n
        "n_cells" -- number of cells fired by an mcParticle\n
        "percentage_fired" -- percentage of cells fired by an mcParticle\n
        "occupancy_per_layer" -- occupancy per layer\n
        "avg_occupancy" -- average occupancy\n
        "occupancies_profile" -- occupancy profile\n
        "cells_per_layer" -- cells per layer\n
    """
    # Create the histogram
    hist = {}
    hist["n_cells"]= []
    hist["percentage_fired"] = []
    hist["occupancy_per_layer"] = []
    hist["avg_occupancy"] = []
    hist["occupancies_profile"] = []
    
    if args == "n_cells" or args == "":
        hist["n_cells"] = dic["list_n_cells_fired_mc"]
        
    if args == "percentage_fired" or args == "":
        hist["percentage_fired"] = dic["percentage_of_fired_cells"]
        
    if args == "occupancy_per_layer" or args == "":
        hist["occupancy_per_layer"] = dic["occupancies_per_layer"]
        
    if args == "avg_occupancy" or args == "":
        hist["avg_occupancy"] = dic["avg_occupancy"]
    
    if args == "occupancies_profile" or args == "":
        hist["occupancies_profile"] = dic["occupancies_per_layer_profile"]
        
    if args == "cells_per_layer":
        hist["n_cells_per_layer"] = dic["n_cell_per_layer"]
        hist["total_number_of_cells"] = dic["total_number_of_cells"]
        hist["total_number_of_layers"] = dic["total_number_of_layers"]
        print(f"total_number_of_cells: {hist['total_number_of_cells']}")
        print(f"total_number_of_layers: {hist['total_number_of_layers']}")
        print(f"n_cells_per_layer: {hist['n_cells_per_layer']}")
    
    return hist
    
    
    

# hist = momPerMC(dic, "")
# hist_plot(hist, imageOutputPath + "momentumMC500.png", "Momentum of MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", xMin=0.0001, xMax=0.175)
# hist_plot(hist, imageOutputPath + "momentumMC500Loggedx.png", "Momentum of MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# hist_plot(hist, imageOutputPath + "momentumMC500Logged.png", "Momentum of MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True, logY=True)

# hist = momPerMC(dic, "onlyOH")
# hist_plot(hist, imageOutputPath + "momentumMC500onlyOneHit.png", "Momentum of MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles",  xMin=0.0001, xMax=0.175)
# hist_plot(hist["onlyOH"], imageOutputPath + "momentumMC500onlyOneHitLoggedx.png", "Momentum of MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# hist_plot(hist, imageOutputPath + "momentumMC500onlyOneHitLogged.png", "Momentum of MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)

# hist = momPerMC(dic, "only+H")
# hist_plot(hist, imageOutputPath + "momentumMC500only+Hit.png", "Momentum of MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles",  xMin=0, xMax=0.3)
# hist_plot(hist["only+H"], imageOutputPath + "momentumMC500only+HitLoggedx.png", "Momentum of MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# hist_plot(hist, imageOutputPath + "momentumMC500only+HitLogged.png", "Momentum of MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)

# hist = momPerMC(dic, "onlyParPhoton")
# hist_plot(hist, imageOutputPath + "momentumMC500onlyParentPhoton.png", "Momentum of MC particles with a Parent Photon (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", xMin=0, xMax=0.12)
# hist_plot(hist, imageOutputPath + "momentumMC500onlyParentPhotonLoggedx.png", "Momentum of MC particles with a Parent Photon (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# hist_plot(hist, imageOutputPath + "momentumMC500onlyParentPhotonLogged.png", "Momentum of MC particles with a Parent Photon (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)

# hist = momPerMC(dic, "ptBelow10R")
# hist_plot(hist, imageOutputPath + "momentumMC500pt10R.png", "Transverse Momentum of MC particles within a radius of 10mm (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", xMin=0, xMax=1)
# hist_plot(hist, imageOutputPath + "momentumMC500pt10RLoggedx.png", "Transverse Momentum of MC particles within a radius of 10mm (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# hist_plot(hist, imageOutputPath + "momentumMC500pt10RLogged.png", "Transverse Momentum of MC particles within a radius of 10mm (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)




####multi:
# hist = PDGPerMC(dic, "")
# bar_plot(hist.keys(), hist.values(), imageOutputPath + "pdgMC500.png", "PDG of MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80)

# hist = PDGPerMC(dic, "electron")
# bar_plot("electron", hist["electron"], imageOutputPath + "pdgElectronPhotonMC500.png", "PDG of MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80, save=False, label="All Electrons")
# bar_plot("electron", hist["e_photon_parent"], imageOutputPath + "pdgElectronPhotonMC500.png", "PDG of MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80, label="Only Electrons with a Parent Photon")

# hist = PDGPerMC(dic, "gen")
# bar_plot("MC Particles", hist["all"], imageOutputPath + "pdgGeneratorStatusMC500.png", "Primary or Secondary MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80, save=False, label="All Particles")
# multi_bar_plot("MC Particles", hist, imageOutputPath + "pdgElectronGeneratorStatusMC500.png", "Primary or Secondary for Electron MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles")

######3maybe return entire dictionary and in plotting remove zero valued keys
# hist = hitsPerMC(dic, "photonSec")
# hist_plot(hist["all"], imageOutputPath + "hitsMC500.png", "Hits of MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Number of MC particles", label="All Particles", autoBin=False, logX=True, logY=True)
# multi_hist_plot(hist, imageOutputPath + "hitsPhotonMC500.png", "Hits of MC particles (" + str(numFiles) + " Files)", xLabel="Number of hits", yLabel="Number of MC particles", label="All Particles", autoBin=True, binLow=0.1, binHigh=900, binSteps=5, binType="lin", logY=True)
# multi_bar_plot("MCP", hist, imageOutputPath + "hitsPhotonMC500.png", "Hits of MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80,  
#          additionalText=f"Percentage of photons that were produced by secondary particles: {round(hist['Only Photons Produced by Secondary Particles']/hist['Only Photons'], 3)}")
# multi_bar_plot("MCP", hist, imageOutputPath + "hitsPDGMC500.png", "Hits of MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80)
# multi_bar_plot("MCP", hist, imageOutputPath + "hitsNeutronMC500.png", "Hits of MC particles (" + str(numFiles) + " Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80,  
#          additionalText=f"Percentage of neutrons that were produced by secondary particles: {round(hist['Only Neutrons Produced by Secondary Particles']/hist['Only Neutrons'], 3)}")

# hist = momPerMC(dic, "")
# hist_plot(hist, imageOutputPath + "momentumMC500.png", "Momentum of MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", xMin=0.0001, xMax=0.175)
# hist_plot(hist, imageOutputPath + "momentumMC500Loggedx.png", "Momentum of MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# hist_plot(hist, imageOutputPath + "momentumMC500Logged.png", "Momentum of MC particles (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True, logY=True)

# hist = momPerMC(dic, "onlyOH", byPDG=True)
#multi_hist_plot(hist, imageOutputPath + "momentumMC500onlyOneHitSepPDG.png", "Momentum of MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles",  xMin=0.0001, xMax=0.175)
# multi_hist_plot(hist, imageOutputPath + "momentumMC500onlyOneHitSepPDGLoggedx.png", "Momentum of MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
#multi_hist_plot(hist, imageOutputPath + "momentumMC500onlyOneHitSepPDGLogged.png", "Momentum of MC particles with only one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)

# hist = momPerMC(dic, "only+H", byPDG=True)
# multi_hist_plot(hist, imageOutputPath + "momentumMC500only+Hit.png", "Momentum of MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles",  xMin=0, xMax=0.3)
# multi_hist_plot(hist, imageOutputPath + "momentumMC500only+HitSetpPDGLoggedx.png", "Momentum of MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# multi_hist_plot(hist, imageOutputPath + "momentumMC500only+HitLogged.png", "Momentum of MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)

# hist = momPerMC(dic, "multiHitsExcludeOne")
# multi_hist_plot(hist, imageOutputPath + "momentumMC500MultiHitsExcludeOneLoggedx.png", "Momentum of MC particles with more than one hit (" + str(numFiles) + " Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)




hist = occupancy(dic, "")
#total number of layers: 112
#total number of cells 56448
#full numpy print
# np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=10)
#print non one values
# print(f"n_cells: {hist['n_cells']}")
# print(f"max: {max(hist['n_cells'])}")
# print(f"size: {len(hist['n_cells'])}")
# print(f"hist['percentage_fired']: {hist['percentage_fired']}")
# print(f"max: {max(hist['percentage_fired'])}")
print(f"hist['avg_occupancy']: {hist['avg_occupancy']}")
# print(f"hist['occupancy_per_layer']: {hist['occupancy_per_layer']}")
# print(f"max: {max(hist['occupancy_per_layer'])}")
hist_plot(hist["n_cells"], imageOutputPath + "occupancyMC500.png", "Occupancy of the detector (" + str(numFiles) + " Files)", xLabel="Number of cells fired by an MC particle", yLabel="Count MC particles", xMin=0, xMax=20, binLow=1, binHigh=86, binSteps=0.3, binType="exp")
# hist_plot(hist["percentage_fired"], imageOutputPath + "occupancyPercMC500.png", "Occupancy of the detector (" + str(numFiles) + " Files)", xLabel="Percentage of cells fired", yLabel="Count MC particles", xMin=0, xMax=80, binHigh=80, binSteps=1, binType="lin")
hist_plot(hist["avg_occupancy"], imageOutputPath + "occupancyPerLayerMC500.png", "Occupancy of the detector (" + str(numFiles) + " Files)", xLabel="Unique Layer Index", yLabel="Occupancy (Count of hits for that ULI)", xMin=0, xMax=110, binHigh=110, binSteps=3, binType="lin")
# x = [int(i) for i in list(hist["n_cells_per_layer"].keys())]
# xy_plot(x, list(hist["n_cells_per_layer"].values()), imageOutputPath + "nCellsPerLayerMC500.png", "Cells Per Layer (" + str(numFiles) + " Files)", xLabel="Number of Layers", yLabel="Cells Per Layer")
# print(f"hist['occupancies_per_layer_profile']: {hist['occupancies_profile']}")
#given hist['occupancies_per_layer_profile'] is an array of tuples that correspond to x and y, plot:
hist_plot(hist['occupancies_profile'][0], imageOutputPath + "occupanciesProfileMC500.png", "Occupancy Profile of the detector (" + str(numFiles) + " Files)", weight=hist['occupancies_profile'][1], bins=hist["occupancies_profile"][0], xLabel="Unique Layer Index", yLabel="Occupancy (Count of hits for that ULI)")




    
    
def genPlot(inputArgs):
    """
    Function that generates the desired histogram.
    Inputs: None
    Outputs: None
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
        
        
        
       
'''
parser = argparse.ArgumentParser()
parser.add_argument('--plot', help="Plot histogram; hist*(str) " +
                    "+ outname*(str) + title(str) + xMin(int) + xMax(int) " +
                    "+ yMin(int) + yMax(int) + xLabel(str) + yLabel(str) " +
                    "+ ylog(bool)", type=str, default="", nargs='9')
args = parser.parse_args()

outputFolder = imageOutputPath + ""

if args.plot:
    try:
        genPlot(args.plot)
        #print(f"Parsed --plot arguments: ...")
    except ValueError as e:
        parser.error(str(e))
#'''
'''    
   
/work/submit/poulin/fccproject-tracking/detector_beam_backgrounds/tracking/trBkgPlt.py --plot

#plot hits per mc
hist hitsMC Number-of-hits-per-MC-particle Number-of-hits Number-of-MC-particles "" "" "" "" Number-of-hits Number-of-MC-particles False


'''