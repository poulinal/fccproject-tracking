import ROOT
import numpy as np 
from utilities.functions import hist_plot, plot_hist, multi_hist_plot, bar_plot
import argparse
import sys
import matplotlib.pyplot as plt
import math


available_functions = ["hitsPerMC", "momPerMC", "pathLenWireMC", "totPathLenMC", "wiresPerMC", "trajLen", "radiusPerMC", "angleHits"]

#open a .npy
dic = np.load("fccproject-tracking/detector_beam_backgrounds/tracking/data/background_particles_500.npy", allow_pickle=True).item()
#dic = np.load("fccproject-tracking/detector_beam_backgrounds/tracking/data/background_particles_20.npy", allow_pickle=True).item()


def hitsPerMC(dic):
    """
    Function that returns the number of hits per MC particle.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the number of hits per MC particle.
    """
    # Create the histogram
    #hist = ROOT.TH1F("hist", "Number of hits per MC particle", 100, 0, 100)
    hist = []
    #set mcParticles to be numpy array size of the last entry of dic["hits"]
    list_index = dic["hits"]
    mcParticles = np.zeros(max(list_index))
    #iterate over all mcParticles and count the number of hits, aka when it repeats in dic["hits"]
    for i in range(0, len(list_index)):
        #print(i)
        mcParticles[list_index[i] - 1] += 1
    #remove all zeros
    mcParticles = mcParticles[mcParticles != 0]
    #iterate over all mcParticles and fill the histogram with the number of hits
    # np.set_printoptions(threshold=sys.maxsize) 
    
    #remove zeros
    mcParticles = mcParticles[mcParticles != 0]
    # print(mcParticles)
    for i in range(0, len(mcParticles)):
        #hist.Fill(mcParticles[i])
        hist.append(mcParticles[i])
    return hist

def momPerMC(dic, args = "", byPDG = False):
    """
    Function that returns the momentum of MC particles.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the momentum of MC particles.
    """
    # Create the histogram
    hist = []
    p = dic["p"]
    
    if byPDG:
        hist = {}
        hist["all"] = []
        pdg = dic["pdg"]
    
    if args == "": #regular get all momenta
        for i in range(0, len(p)):
            hist.append(p[i])
        # print(f"max hist: {max(hist)}")
        return hist


    if args == "onlyOH" or args == "only+H":
        count_hits = dic["count_hits"] #the index of the mcParticle for each hit
        oneHit = []
        multHit = []
        #seperate hits based on if they occur once or more than once with the same mcParticle
        for i in range(0, len(count_hits)):
            if count_hits[i] == 1:
                oneHit.append(i)
                if args == "onlyOH":
                    if byPDG:
                        if pdg[i] in hist:
                            hist[pdg[i]].append(p[i])
                        else:
                            hist[pdg[i]] = [p[i]]
                    else:
                        hist.append(p[i])
                    hist["all"].append(p[i])
            else:
                multHit.append(i)
                if args == "only+H":
                    if byPDG:
                        if pdg[i] in hist:
                            hist[pdg[i]].append(p[i])
                        else:
                            hist[pdg[i]] = [p[i]]
                    else:
                        hist.append(p[i])
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
                
def PDGPerMC(dic, args = ""):
    """
    Function that returns the PDG of MC particles.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the PDG of MC particles.
    """
    # Create the histogram
    pdg = dic["pdg"]
    #get unique pdg's
    pdg_unique = np.unique(pdg)
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
        Return: return_description
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
        
        for i in range(0, len(gen)):
            hist_dic["all"] += 1
            if gen[i] == 0:
                hist_dic["primary"] += 1
            if gen[i] == 1:
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


# hist = momPerMC(dic, "")
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500.png", "Momentum of MC particles (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", xMin=0.0001, xMax=0.175)
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500Loggedx.png", "Momentum of MC particles (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500Logged.png", "Momentum of MC particles (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True, logY=True)

# hist = momPerMC(dic, "onlyOH")
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500onlyOneHit.png", "Momentum of MC particles with only one hit (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles",  xMin=0.0001, xMax=0.175)
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500onlyOneHitLoggedx.png", "Momentum of MC particles with only one hit (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500onlyOneHitLogged.png", "Momentum of MC particles with only one hit (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)

# hist = momPerMC(dic, "only+H")
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500only+Hit.png", "Momentum of MC particles with more than one hit (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles",  xMin=0, xMax=0.3)
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500only+HitLoggedx.png", "Momentum of MC particles with more than one hit (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500only+HitLogged.png", "Momentum of MC particles with more than one hit (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)

# hist = momPerMC(dic, "onlyParPhoton")
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500onlyParentPhoton.png", "Momentum of MC particles with a Parent Photon (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", xMin=0, xMax=0.12)
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500onlyParentPhotonLoggedx.png", "Momentum of MC particles with a Parent Photon (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500onlyParentPhotonLogged.png", "Momentum of MC particles with a Parent Photon (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)

# hist = momPerMC(dic, "ptBelow10R")
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500pt10R.png", "Transverse Momentum of MC particles within a radius of 10mm (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", xMin=0, xMax=1)
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500pt10RLoggedx.png", "Transverse Momentum of MC particles within a radius of 10mm (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500pt10RLogged.png", "Transverse Momentum of MC particles within a radius of 10mm (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)




####multi:
# hist = PDGPerMC(dic, "")
# bar_plot(hist.keys(), hist.values(), "fccproject-tracking/detector_beam_backgrounds/tracking/images/pdgMC500.png", "PDG of MC particles (500 Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80)

# hist = PDGPerMC(dic, "electron")
# bar_plot("electron", hist["electron"], "fccproject-tracking/detector_beam_backgrounds/tracking/images/pdgElectronPhotonMC500.png", "PDG of MC particles (500 Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80, save=False, label="All Electrons")
# bar_plot("electron", hist["e_photon_parent"], "fccproject-tracking/detector_beam_backgrounds/tracking/images/pdgElectronPhotonMC500.png", "PDG of MC particles (500 Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80, label="Only Electrons with a Parent Photon")

# hist = PDGPerMC(dic, "gen")
# bar_plot("MC Particles", hist["all"], "fccproject-tracking/detector_beam_backgrounds/tracking/images/pdgGeneratorStatusMC500.png", "Primary or Secondary MC particles (500 Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80, save=False, label="All Particles")
# bar_plot("MC Particles", hist["primary"], "fccproject-tracking/detector_beam_backgrounds/tracking/images/pdgGeneratorStatusMC500.png", "Primary or Secondary MC particles (500 Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80, save=False, label="Only Primary Particles")
# bar_plot("MC Particles", hist["secondary"], "fccproject-tracking/detector_beam_backgrounds/tracking/images/pdgGeneratorStatusMC500.png", "Primary or Secondary MC particles (500 Files)", xLabel="PDG", yLabel="Count MC particles", rotation=80, label="Only Secondary Particles")




# hist = momPerMC(dic, "")
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500.png", "Momentum of MC particles (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", xMin=0.0001, xMax=0.175)
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500Loggedx.png", "Momentum of MC particles (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500Logged.png", "Momentum of MC particles (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True, logY=True)

# hist = momPerMC(dic, "onlyOH", byPDG=True)
#multi_hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500onlyOneHitSepPDG.png", "Momentum of MC particles with only one hit (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles",  xMin=0.0001, xMax=0.175)
# multi_hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500onlyOneHitSepPDGLoggedx.png", "Momentum of MC particles with only one hit (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
#multi_hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500onlyOneHitSepPDGLogged.png", "Momentum of MC particles with only one hit (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)

hist = momPerMC(dic, "only+H", byPDG=True)
# multi_hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500only+Hit.png", "Momentum of MC particles with more than one hit (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles",  xMin=0, xMax=0.3)
multi_hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500only+HitSetpPDGLoggedx.png", "Momentum of MC particles with more than one hit (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logX=True)
# multi_hist_plot(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500only+HitLogged.png", "Momentum of MC particles with more than one hit (500 Files)", xLabel="Momentum (GeV)", yLabel="Count MC particles", logY=True, logX=True)











    
    
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
            plot_hist(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/" + outname + ".png", title, xLabel, yLabel, ylog)
    if emY:
        plot_hist(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/" + outname + ".png", title, xMin, xMax, yLabel, ylog)
    else:
        plot_hist(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/" + outname + ".png", title, xMin, xMax, yMin, yMax, xLabel, yLabel, ylog)
        
        
        
       
'''
parser = argparse.ArgumentParser()
parser.add_argument('--plot', help="Plot histogram; hist*(str) " +
                    "+ outname*(str) + title(str) + xMin(int) + xMax(int) " +
                    "+ yMin(int) + yMax(int) + xLabel(str) + yLabel(str) " +
                    "+ ylog(bool)", type=str, default="", nargs='9')
args = parser.parse_args()

outputFolder = "fccproject-tracking/detector_beam_backgrounds/tracking/images/"

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