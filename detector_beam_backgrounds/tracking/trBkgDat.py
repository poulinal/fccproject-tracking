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
for i in range(1,20):
    list_overlay.append("/ceph/submit/data/group/fcc/ee/detector/tracking/IDEA_background_only/out_sim_edm4hep_background_"+str(i)+".root")

dic = {}
total_time = 0 
rootfile = list_overlay[0]

reader = root_io.Reader(rootfile)
metadata = reader.get("metadata")[0]
cellid_encoding = metadata.get_parameter("CDCHHits__CellIDEncoding")
decoder = dd4hep.BitFieldCoder(cellid_encoding)

list_index = []
list_hit_path_length = []
list_pos_hit = []
list_superLayer = []
list_layer = []
list_phi = []
list_stereo = []
list_pos_z = []
for event in reader.get("events"):
    dc_hits = event.get("CDCHHits")
    for num_hit, dc_hit in enumerate(dc_hits):
        mcParticle = dc_hit.getMCParticle()
        index_mc = mcParticle.getObjectID().index
        list_index.append(index_mc)
        list_hit_path_length.append(dc_hit.getPathLength())
        list_pos_hit.append([dc_hit.getPosition().x, dc_hit.getPosition().y, dc_hit.getPosition().z])
        list_pos_z.append(dc_hit.getPosition().z)
        #print(list_index)
        
        cellID = dc_hit.getCellID()
        list_superLayer.append(decoder.get(cellID, "superLayer"))
        list_layer.append(decoder.get(cellID, "layer"))
        list_phi.append(decoder.get(cellID, "phi"))
        list_stereo.append(decoder.get(cellID, "stereo"))
        
unique_mcs = np.unique(np.array(list_index))
print("unique_mcs", unique_mcs)
list_R = []
list_p = []
list_px = []
list_py = []
list_pz = []
list_gen_status = []

#list_path_length = []
list_pos_ver = []

MCparticles = event.get("MCParticles")
for i in range(0, len(unique_mcs)):
    mc_index = unique_mcs[i]
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
    
    #pathLength = mcParticle.getPathLength()
    list_pos_ver.append([x_vertex, y_vertex, z_vertex])
    
    
#from mcParticle
dic["R"]=np.array(list_R)
dic["p"]=np.array(list_p)
dic["px"]=np.array(list_px)
dic["py"]=np.array(list_py)
dic["pz"]=np.array(list_pz)
dic["gens"]=np.array(list_gen_status)
dic["pos_ver"] = np.array(list_pos_ver)

#from hits
dic["hits"]=np.array(list_index)
dic["hits_pL"]=np.array(list_hit_path_length)
#dic["pL"] = np.array(list_path_length)
dic["pos_hit"] = np.array(list_pos_hit)
dic["unique_mcs"] = unique_mcs
dic["superLayer"] = np.array(list_superLayer)
dic["layer"] = np.array(list_layer)
dic["phi"] = np.array(list_phi)
dic["stereo"] = np.array(list_stereo)
dic["pos_z"] = np.array(list_pos_z)


#print(f"list_hit_path_length: {list_hit_path_length}")
#print(f"allHits: {len(list_hit_path_length)}")
#print(f"unique_mcs: {len(unique_mcs)}")
#print(f"lenPathLength: {len(list_hit_path_length)}")
#print(f"pos_hit: {list_pos_hit}")
###path length is for each hit not combined for each particle
#print(list_path_length)

np.save("fccproject-tracking/detector_beam_backgrounds/tracking/data/background_particles.npy", dic)