import ROOT
import numpy as np 
from utilities.functions import hist_plot, plot_hist#, sns_plot
import argparse
import sys
import matplotlib.pyplot as plt
import math


available_functions = ["hitsPerMC", "momPerMC", "pathLenWireMC", "totPathLenMC", "wiresPerMC", "trajLen", "radiusPerMC", "angleHits"]

#open a .npy
#dic = np.load("fccproject-tracking/detector_beam_backgrounds/tracking/data/background_particles_500.npy", allow_pickle=True).item()
dic = np.load("fccproject-tracking/detector_beam_backgrounds/tracking/data/background_particles_20.npy", allow_pickle=True).item()


def hitsPerMC(dic):
    """
    Function that returns the number of hits per MC particle.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the number of hits per MC particle.
    """
    # Create the histogram
    hist = ROOT.TH1F("hist", "Number of hits per MC particle", 100, 0, 100)
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
    np.set_printoptions(threshold=sys.maxsize) 
    
    #remove zeros
    mcParticles = mcParticles[mcParticles != 0]
    print(mcParticles)
    for i in range(0, len(mcParticles)):
        hist.Fill(mcParticles[i])
    return hist

def momPerMC(dic, args = ""):
    """
    Function that returns the momentum of MC particles.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the momentum of MC particles.
    """
    # Create the histogram
    hist = ROOT.TH1F("hist", "Momentum of MC particles", 100, 0, 1)
    p = dic["p"]
    print(max(p))
    if args == "":
        # Fill the histogram
        for i in range(0, len(p)):
            hist.Fill(p[i])
        return hist


    if args == "onlyOH" or args == "only+H":
        hits = dic["hits"] #the index of the mcParticle for each hit
        
        oneHit = []
        multHit = []
        #seperate hits based on if they occur once or more than once with the same mcParticle
        for i in range(0, len(hits)):
            if hits.count(hits[i]) == 1:
                oneHit.append(i)
            else:
                multHit.append(i)
                
        if args == "onlyOH":
            for i in range(0,oneHit):
                hist.Fill(p[i])
        
        if args == "only+H":
            for i in range(0,multHit):
                hist.Fill(p[i])
    
    if args == "onlylow":
        p = dic["p"]
        p = p[p < 1]
        return p
    
    if args == "parPhoton":
        """
        If the mcParticle has a parent that is a photon, return the momentum of the mcParticle.
        Keyword arguments:
        argument -- description
        """
        photon = dic["bPhoton"] #list t or false if mcParticle has a parent thats a photon
        for i in range(0, len(photon)):
            if photon[i]:
                hist.Fill(p[i])
    
    if args == "below10R":
        """
        If the mcParticle's vertex radius is below 10, fill the histogram with the transverse momentum of the mcParticle.
        Keyword arguments:
        argument -- description
        """
        R = dic["R"]
        px = dic["px"]
        py = dic["py"]
        for i in range(0, len(R)):
            if R[i] < 10:
                hist.Fill(math.sqrt(px[i]**2 + py[i]**2))
                
def PDGPerMC(dic, args = ""):
    """
    Function that returns the PDG of MC particles.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the PDG of MC particles.
    """
    # Create the histogram
    pdg = dic["PDG"]
    #get unique pdg's
    pdg_unique = np.unique(pdg)
    hist = ROOT.TH1F("hist", "PDG of MC particles", pdg_unique, 0, max(pdg_unique))
    if args == "":
        for i in range(0, len(pdg)):
            hist.Fill(pdg[i])
        return hist
    if args == "photon":
        for i in range(0, len(pdg)):
            if pdg[i] == 22:
                hist.Fill(pdg[i])
        return hist
    if args == "electron":
        for i in range(0, len(pdg)):
            if pdg[i] == 11:
                hist.Fill(pdg[i])
        return hist
    if args == "gen":
        gen = dic["gens"]
        for i in range(0, len(gen)):
            if gen[i] == 1:
                hist.Fill(pdg[i])
        return hist


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
    hist = ROOT.TH1F("hist", "Number of wires per MC particle", 100, 0, 100)
    # Fill the histogram
    list_index = dic["hits"]
    mcParticles = np.zeros(max(list_index))
    for i in range(0, len(list_index)):
        mcParticles[list_index[i] - 1] += 1
    mcParticles = mcParticles[mcParticles != 0]
    for i in range(0, len(mcParticles)):
        hist.Fill(mcParticles[i])
    return hist

def trajLen(dic):
    """
    Function that returns the trajectory length of MC particles.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the trajectory length of MC particles.
    """
    hist = ROOT.TH1F("hist", "Trajectory length of MC particles", 100, 0, 4000)
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
        hist.Fill(totMCPL[i])
    return hist, totMCPL

def radiusPerMC(dic):
    """
    Function that returns the radius of MC particles.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the radius of MC particles.
    """
    # Create the histogram
    hist = ROOT.TH1F("hist", "Radius of MC particles", 100, 0, 5)
    #mcParticleR = groupHits(dic, dic["R"])
    # Fill the histogram
    #print(max(dic["R"]))
    for i in range(0, len(dic["R"])):
        hist.Fill(dic["R"][i])
    return hist

def angleHits(dic):
    """
    Function that returns the angle of hits.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the angle of hits.
    """
    # Create the histogram
    hist = ROOT.TH1F("hist", "Angle of hits", 100, 0, 2*math.pi)
    # Fill the histogram
    for i in range(0, len(dic["px"])):
        hist.Fill(phi(dic["px"][i], dic["py"][i]))
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


#'''
hist = momPerMC(dic)
#print(dic["p"])
#plot_hist(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC500.png", "Momentum of MC particles (500 Files)", xLabel="Momentum (GeV)", yLabel="Number of MC particles", logY=True)
#plot_hist(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMCZoomed500.png", "Momentum of MC particles (500 Files)", xLabel="Momentum (GeV)", yLabel="Number of MC particles", xMin=0, xMax=2, logY=True)

plot_hist(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC20.png", "Momentum of MC particles (20 Files)", xLabel="Momentum (GeV)", yLabel="Number of MC particles", logY=True)
#plot_hist(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMCZoomed20.png", "Momentum of MC particles (20 Files)", xLabel="Momentum (GeV)", yLabel="Number of MC particles", xMin=0, xMax=1, logY=True)
#'''

'''
hist = hitsPerMC(dic)
#print(hist)
plot_hist(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/hitsMC.png", "Number of hits per MC particle", xLabel="Number of hits", yLabel="Number of MC Particles", logY=True)
#sns_plot(dic["p"], "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMC.png", title="Momentum of MC particles", xMin="Momentum (GeV)", xMax="Number of MC particles", yLabel=True)
#'''

'''
hist, totMCPL = trajLen(dic)


plot_hist(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/trajLengthMC.png", "Trajectory length of MC particles", yLabel="Number of MC particles", xLabel="Trajectory length (mm)", logY=True)
#'''

'''
hist = radiusPerMC(dic)
plot_hist(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/radiusMC.png", "Radius of MC particles", yLabel="Number of MC particles", xLabel="Radius(mm)", logY=True)
'''











    
    
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