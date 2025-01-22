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

list_overlay = []
numfiles = 500
for i in range(1,numfiles + 1):
    list_overlay.append("/ceph/submit/data/group/fcc/ee/detector/tracking/IDEA_background_only/out_sim_edm4hep_background_"+str(i)+".root")

dic = {}
dic_file_path = "fccproject-tracking/detector_beam_backgrounds/tracking/data/background_particles_500.npy"
#assign dic to empty dictionary
dic = {}
dic["R"] = []
dic["p"] = []
dic["px"] = []
dic["py"] = []
dic["pz"] = []
dic["gens"] = []
dic["pos_ver"] = []
dic["hits"] = []
dic["hits_pL"] = []
dic["pos_hit"] = []
dic["unique_mcs"] = []
dic["superLayer"] = []
dic["layer"] = []
dic["phi"] = []
dic["stereo"] = []
dic["pos_z"] = []
dic["count_hits"] = []
dic["has_par_photon"] = []
dic["pdg"] = []

np.save(dic_file_path, dic)

#loop over all the files
for i in range(0, len(list_overlay)):
    dic = np.load(dic_file_path, allow_pickle=True).item()  
    list_index = []
    list_hit_path_length = []
    list_pos_hit = []
    list_superLayer = []
    list_layer = []
    list_phi = []
    list_stereo = []
    list_pos_z = []

    list_R = []
    list_p = []
    list_px = []
    list_py = []
    list_pz = []
    list_gen_status = []
    #list_path_length = []
    list_pos_ver = []
    list_par_photon = []
    list_pdg = []
    
    seen = []
    count_hits = []

    rootfile = list_overlay[i]
    print(f"Running over file: {rootfile}")

    reader = root_io.Reader(rootfile)
    metadata = reader.get("metadata")[0]
    cellid_encoding = metadata.get_parameter("CDCHHits__CellIDEncoding")
    decoder = dd4hep.BitFieldCoder(cellid_encoding)

    for event in reader.get("events"):
        dc_hits = event.get("CDCHHits")
        for num_hit, dc_hit in enumerate(dc_hits):
            mcParticle = dc_hit.getMCParticle()
            index_mc = mcParticle.getObjectID().index
            list_index.append(index_mc)
            #list_hit_path_length.append(dc_hit.getPathLength())
            list_pos_hit.append([dc_hit.getPosition().x, dc_hit.getPosition().y, dc_hit.getPosition().z])
            list_pos_z.append(dc_hit.getPosition().z)
            #print(list_index)
            
            #check if the hit was produced by a particle already seen (in list_index)
            if index_mc not in seen:
                #add 1 to the count of hits at this index
                count_hits.append(1)
                seen.append(index_mc)
            else:
                #find the index of the hit in seen
                index = seen.index(index_mc)
                #add 1 to the count of hits at this index
                count_hits[index] += 1
            
            cellID = dc_hit.getCellID()
            list_superLayer.append(decoder.get(cellID, "superLayer"))
            list_layer.append(decoder.get(cellID, "layer"))
            list_phi.append(decoder.get(cellID, "phi"))
            list_stereo.append(decoder.get(cellID, "stereo"))

    unique_mcs = np.unique(np.array(list_index))
        
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
        #pathLength = mcParticle.getPathLength()
        #list_pos_ver.append([x_vertex, y_vertex, z_vertex])
    #print(f"len(unique_mcs): {len(unique_mcs)}")
    #print(f"len(p): {len(list_p)}")
    
    #update the dictionary (append)
    #print(dic["R"])
    dic["R"] = np.append(dic["R"], np.array(list_R))
    dic["p"] = np.append(dic["p"], np.array(list_p))
    dic["px"] = np.append(dic["px"], np.array(list_px))
    dic["py"] = np.append(dic["py"], np.array(list_py))
    dic["pz"] = np.append(dic["pz"], np.array(list_pz))
    dic["gens"] = np.append(dic["gens"], np.array(list_gen_status))
    dic["pos_ver"] = np.append(dic["pos_ver"], np.array(list_pos_ver))
    
    dic["hits"] = np.append(dic["hits"], np.array(list_index))
    #dic["hits_pL"] = np.append(dic["hits_pL"], np.array(list_hit_path_length))
    dic["pos_hit"] = np.append(dic["pos_hit"], np.array(list_pos_hit))
    dic["unique_mcs"] = np.append(dic["unique_mcs"], np.array(unique_mcs))
    dic["superLayer"] = np.append(dic["superLayer"], np.array(list_superLayer))
    dic["layer"] = np.append(dic["layer"], np.array(list_layer))
    dic["phi"] = np.append(dic["phi"], np.array(list_phi))
    dic["stereo"] = np.append(dic["stereo"], np.array(list_stereo))
    dic["pos_z"] = np.append(dic["pos_z"], np.array(list_pos_z))
    dic["count_hits"] = np.append(dic["count_hits"], np.array(count_hits))
    dic["has_par_photon"] = np.append(dic["has_par_photon"], np.array(list_par_photon))
    dic["pdg"] = np.append(dic["pdg"], np.array(list_pdg))
    
        
    np.save(dic_file_path, dic)
