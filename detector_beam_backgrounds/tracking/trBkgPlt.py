import ROOT
import numpy as np 
from utilities.functions import hist_plot, plot_hist#, sns_plot
import argparse
import sys
import matplotlib.pyplot as plt
import math


available_functions = ["hitsPerMC", "momPerMC", "pathLenWireMC", "totPathLenMC", "wiresPerMC", "trajLen", "radiusPerMC", "angleHits"]

#open a .npy
dic = np.load("fccproject-tracking/detector_beam_backgrounds/tracking/data/background_particles.npy", allow_pickle=True).item()

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

def momPerMC(dic):
    """
    Function that returns the momentum of MC particles.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the momentum of MC particles.
    """
    # Create the histogram
    hist = ROOT.TH1F("hist", "Momentum of MC particles", 100, 0, 1e-02)
    # Fill the histogram
    for i in range(0, len(dic["p"])):
        #print(dic["p"][i])
        hist.Fill(dic["p"][i])
    return hist

def pathLenWireMC(dic, combined=False):
    """
    Function that returns the path length of MC particles.
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the path length of MC particles.
    """
    # Create the histogram
    #hist = ROOT.TH1F("hist", "Path length of MC particles", 1000, 0, 1500)
    # Fill the histogram
    ###this is for each hit, not combined for each particle####
    
    if combined:
        hist = ROOT.TH1F("hist", "Path length of MC particles", 500, 0, 1000)
        pathLen = derPathLenWireEachMC(dic)
        for i in range(0, len(pathLen)):
            hist.Fill(pathLen[i])
    
    else:
        hist = ROOT.TH1F("hist", "Path length of MC particles", 1000, 0, 1500)
        print(f"len(dic['hits_pL']): {len(dic['hits_pL'])}")
        for i in range(0, len(dic["hits_pL"])):
            hist.Fill(dic["hits_pL"][i])
    return hist

def derPathLenWireEachMC(dic):
    """
    Function that iterates through the path length hits
    and combine the path length for each MC particle.
    This value is the total length of the MC particle along the wire
    Keyword arguments:
    argument -- dictionary which has the keys R, p, px, py, pz, gens, hits_pL, pos_ver, pos_hit...
    Return: an array where each index has the path length for the corresponding MC particle.
    """
    list_hit_path_length = dic["hits_pL"]
    list_index = dic["hits"]
    mcParticlesPL = np.zeros(max(list_index))
    for i in range(0, len(list_index)):
        #print(i)
        mcParticlesPL[list_index[i] - 1] += list_hit_path_length[i]
        
    #remove all zeros
    mcParticlesPL = mcParticlesPL[mcParticlesPL != 0]
    print(f"mcParticlesPL: {mcParticlesPL}")
    print(f"len(mcParticlesPL): {len(mcParticlesPL)}")
    return mcParticlesPL

def totPathLenMC(dic):
    """
    Function that returns the total path length of MC particles.,
    aka the sum of the vertex to the first hit and the sum of the path length
    along the wire
    Inputs: dic, dictionary with keys R, p, px, py, pz, gens.
    Outputs: hist, histogram with the total path length of MC particles.
    """
    # Create the histogram
    #hist = ROOT.TH1F("hist", "Total path length of MC particles", 100, 0, 1500)
    
    #get the length of the vertex to the first hit
    mcParticlesPosHit = groupHits(dic, "pos_hit")
    #mcParticlesPL = derPathLenWireEachMC(dic)
    mcParticlesPosVer = dic["pos_ver"]
    totMCPL = []
    #go through each mcParticle, 
    # make a sum of the distance from the vertex to the first hit
    #and the sum of the path length along the wire
    #where mcParticlesPosVer is a 3position array 
    # and mcParticlesPosHit is an array (each particle) of 3position arrays (each hit)
    for i in range(0, len(mcParticlesPosHit)):
        totPL = np.linalg.norm(sum(mcParticlesPosHit[i])) + np.linalg.norm(mcParticlesPosHit[i][0] - mcParticlesPosVer[i])
        totMCPL.append(totPL)
    print(f"totMCPL: {len(totMCPL)}")
    print(f"totMCPL max: {max(totMCPL)}")
    print(f"totMCPL min: {min(totMCPL)}")
    print(f"totmcpl dif: {max(totMCPL) - min(totMCPL)}")
    hist = ROOT.TH1F("hist", "Total path length of MC particles", 594, 0, 594)
    for i in range(0, len(totMCPL)):
        hist.Fill(totMCPL[i])
    return hist, totMCPL


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


'''
hist = momPerMC(dic)
print(dic["p"])
plot_hist(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/momentumMCZoomed.png", "Momentum of MC particles", xLabel="Momentum (GeV)", yLabel="Number of MC particles", xMin=0, xMax=1e-02, logY=True)
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

    #check if inputArgs[2] type
    if type(inputArgs[2]) == str:
        title = inputArgs[2]
    elif type(inputArgs[2]) == str:
        title = ""
        xMin = inputArgs[2]
        
        
    xMin = float(inputArgs[3])  # Integer
    xMax = float(inputArgs[4])  # Integer
    yMin = float(inputArgs[5])  # Integer
    yMax = float(inputArgs[6])  # Integer
    xLabel = inputArgs[7]  # String
    yLabel = inputArgs[8]  # String
    ylog = bool(inputArgs[9])  # Boolean
    
    plot_hist(hist, "fccproject-tracking/detector_beam_backgrounds/tracking/images/" + outname + ".png", title, xMin, xMax, yMin, yMax, xLabel, yLabel, ylog)
        
        
        
        
parser = argparse.ArgumentParser()
parser.add_argument('--plot', help="Plot histogram; hist*(str) " +
                    "+ outname*(str) + title(str) + xMin(int) + xMax(int) " +
                    "+ yMin(int) + yMax(int) + xLabel(str) + yLabel(str) " +
                    "+ ylog(bool)", type=str, default="", nargs='+')
args = parser.parse_args()

outputFolder = "fccproject-tracking/detector_beam_backgrounds/tracking/images/"

if args.plot:
    try:
        genPlot(args.plot)
        #print(f"Parsed --plot arguments: ...")
    except ValueError as e:
        parser.error(str(e))
        
'''    
   
/work/submit/poulin/fccproject-tracking/detector_beam_backgrounds/tracking/trBkgPlt.py --plot

#plot hits per mc
hist hitsMC Number-of-hits-per-MC-particle Number-of-hits Number-of-MC-particles


'''