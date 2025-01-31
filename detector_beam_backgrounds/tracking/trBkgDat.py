from podio import root_io
#import edm4hep
#import sys
#import ROOT
#from ROOT import TFile, TTree
import numpy as np 
#from array import array
import math
#import os
import dd4hep as dd4hepModule
from ROOT import dd4hep
import sys


list_overlay = []
numfiles = 500

# oldDataPath = "/ceph/submit/data/group/fcc/ee/detector/tracking/IDEA_background_only/"
bkgDataPath = "/ceph/submit/data/group/fcc/ee/detector/tracking/IDEA_background_only_IDEA_o1_v03_v3/"
# bkgDataPath = "/ceph/submit/data/group/fcc/ee/detector/tracking/IDEA_background_only_IDEA_o1_v03_v1/"
# combinedDataPath = "/ceph/submit/data/group/fcc/ee/detector/tracking/Zcard_CLD_background_v1/"
combinedDataPath = "/ceph/submit/data/group/fcc/ee/detector/tracking/Zcard_CLD_background_IDEA_o1_v03_v4/Zcard_CLD_background_IDEA_o1_v03_v4/"
bkgFilePath = "out_sim_edm4hep_background_"
combinedFilePath = "/out_sim_edm4hep"
signalFilePath = "/out_sim_edm4hep_base"
type = "bkg"
print("type: ", type)
input("Press Enter to continue...")


for i in range(1,numfiles + 1):
    #list_overlay.append(oldDataPath + "out_sim_edm4hep_background_"+str(i)+".root")
    if type == "combined":
        list_overlay.append(combinedDataPath + str(i) + combinedFilePath + ".root")
    elif type == "bkg":
        list_overlay.append(bkgDataPath + bkgFilePath +str(i)+".root")
    elif type == "signal":
        list_overlay.append(combinedDataPath + str(i) + signalFilePath + ".root")
    else:
        #throw error
        print("Error: type not recognized")
        sys.exit(1)

dic = {}
dic_file_path = "fccproject-tracking/detector_beam_backgrounds/tracking/data/occupancy_tinker/" + str(type) + "_background_particles_" + str(numfiles) + ".npy"

keys = ["R", "p", "px", "py", "pz", "gens", "pos_ver", "hits", 
        "pos_hit", "unique_mcs", "superLayer", "layer", "nphi", "stereo", 
        "pos_z", "count_hits", "has_par_photon", "pdg", 
        "hits_produced_secondary", "hits_mc_produced_secondary"]
#assign dic to empty dictionary
dic = {}
for key in keys:
    dic[key] = []
    
np.save(dic_file_path, dic)
    
dic = np.load(dic_file_path, allow_pickle=True).item() 
    



#loop over all the files
for i in range(0, len(list_overlay)): 

    rootfile = list_overlay[i]
    print(f"Running over file: {rootfile}")
    reader = root_io.Reader(rootfile)
    metadata = reader.get("metadata")[0]
    if type == "":
        cellid_encoding = metadata.get_parameter("CDCHHits__CellIDEncoding")
    else:
        cellid_encoding = metadata.get_parameter("DCHCollection__CellIDEncoding")
    decoder = dd4hep.BitFieldCoder(cellid_encoding)
    
    numEvents = 0
    for event in reader.get("events"):
        numEvents += 1
        list_index, list_hit_path_length, list_pos_hit, list_superLayer, \
        list_layer, list_stereo, list_nphi, list_pos_z = [], [], [], [], [], [], [], []
        list_R, list_p, list_px, list_py, list_pz, list_gen_status, \
            list_pos_ver, list_par_photon, list_pdg, \
                list_hits_secondary, list_hits_mc_secondary,\
                    percentage_of_fired_cells= [], [], [], [], [], [], [], [], [], [], [], []
        seen, count_hits = [], []
    
        if type == "":
            dc_hits = event.get("CDCHHits")
        else:
            dc_hits = event.get("DCHCollection")
        
        # print("starting hits")
        for num_hit, dc_hit in enumerate(dc_hits):
            mcParticle = dc_hit.getMCParticle()
            index_mc = mcParticle.getObjectID().index
            list_index.append(index_mc)
            list_pos_hit.append([dc_hit.getPosition().x, dc_hit.getPosition().y, dc_hit.getPosition().z])
            list_pos_z.append(dc_hit.getPosition().z)
            
            #check if the hit was produced by a particle already seen (in list_index)
            if index_mc not in seen:
                #add 1 to the count of hits at this index
                count_hits.append(1)
                list_hits_mc_secondary.append(dc_hit.isProducedBySecondary())
                seen.append(index_mc)
            else:
                #find the index of the hit in seen
                index = seen.index(index_mc)
                #add 1 to the count of hits at this index
                count_hits[index] += 1
                list_hits_mc_secondary[index] = max(list_hits_mc_secondary[index], dc_hit.isProducedBySecondary())
                
            list_hits_secondary.append(dc_hit.isProducedBySecondary())
        #end of hit loop
    

        unique_mcs = np.unique(np.array(list_index))
        #get mcParticle data
        MCparticles = event.get("MCParticles")
        for j in range(0, len(unique_mcs)):
            mc_index = unique_mcs[j]
            mcParticle = MCparticles[int(mc_index)]
            
            x_vertex = mcParticle.getVertex().x
            y_vertex = mcParticle.getVertex().y
            z_vertex = mcParticle.getVertex().z
            vertex_R = math.sqrt(mcParticle.getVertex().x ** 2 + mcParticle.getVertex().y ** 2)* 1e-03
            list_R.append(vertex_R)
            momentum = mcParticle.getMomentum()
            p = math.sqrt(momentum.x**2 + momentum.y**2 + momentum.z**2)
            list_p.append(p)
            list_px.append(momentum.x)
            list_py.append(momentum.y)
            list_pz.append(momentum.z)
            gen_status = mcParticle.getGeneratorStatus()
            list_gen_status.append(gen_status)
            
            list_pdg.append(mcParticle.getPDG())
            has_photon_parent = 0
            for parent in mcParticle.getParents():
                if parent.getPDG() == 22:
                    has_photon_parent = 1
            list_par_photon.append(has_photon_parent) 
        

        
        print("updating dictionary")
        #update dictionary
        dic["R"] += (list_R)
        dic["p"] += (list_p)
        dic["px"] += (list_px)
        dic["py"] += (list_py)
        dic["pz"] += (list_pz)
        dic["gens"] += (list_gen_status)
        dic["pos_ver"] += (list_pos_ver)

        dic["hits"] += (list_index)
        dic["pos_hit"] += (list_pos_hit)
        dic["unique_mcs"] += (unique_mcs.tolist())
        dic["superLayer"] += (list_superLayer)
        dic["layer"] += (list_layer)
        dic["nphi"] += (list_nphi)
        dic["stereo"] += (list_stereo)
        dic["pos_z"] += (list_pos_z)
        dic["count_hits"] += (count_hits)
        dic["has_par_photon"] += (list_par_photon)
        dic["pdg"] += (list_pdg)
        dic["hits_produced_secondary"] += (list_hits_secondary)
        dic["hits_mc_produced_secondary"] += (list_hits_mc_secondary)
        
    #end of event loop
                
                
#end of file loop

print("end of file loop")


print(f"Saving dictionary to {dic_file_path}")
np.save(dic_file_path, dic)
