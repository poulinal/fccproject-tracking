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
numfiles = 20

# oldDataPath = "/ceph/submit/data/group/fcc/ee/detector/tracking/IDEA_background_only/"
bkgDataPath = "/ceph/submit/data/group/fcc/ee/detector/tracking/IDEA_background_only_IDEA_o1_v03_v3/"
# combinedDataPath = "/ceph/submit/data/group/fcc/ee/detector/tracking/Zcard_CLD_background_v1/"
combinedDataPath = "/ceph/submit/data/group/fcc/ee/detector/tracking/Zcard_CLD_background_IDEA_o1_v03_v4/Zcard_CLD_background_IDEA_o1_v03_v4/"
bkgFilePath = "out_sim_edm4hep_background_"
combinedFilePath = "/out_sim_edm4hep"
signalFilePath = "/out_sim_edm4hep_base"
type = "bkg"

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
dic_file_path = "fccproject-tracking/detector_beam_backgrounds/tracking/data/" + str(type) + "_background_particles_" + str(numfiles) + ".npy"

keys = ["R", "p", "px", "py", "pz", "gens", "pos_ver", "hits", 
        "pos_hit", "unique_mcs", "superLayer", "layer", "nphi", "stereo", 
        "pos_z", "count_hits", "has_par_photon", "pdg", 
        "hits_produced_secondary", "hits_mc_produced_secondary", 
        "percentage_of_fired_cells", "list_n_cells_fired_mc", "occupancies_per_layer",
        "avg_occupancy", "occupancies_per_layer_profile", "dict_layer_phiSet",
        "n_cell_per_layer", "total_number_of_cells", "total_number_of_layers", "max_n_cell_per_layer",]
#assign dic to empty dictionary
dic = {}
for key in keys:
    dic[key] = []
    #note that hits_mc_produced_secondary is for every mc which produced a hit
    #hits_produced_secondary is for every hit, was it produced by a secondary particle?
    
# Drift chamber geometry parameters
n_layers_per_superlayer = 8
n_superlayers = 14
total_number_of_layers = 0
n_cell_superlayer0 = 192
n_cell_increment = 48
n_cell_per_layer = {}
total_number_of_cells = 0
for sl in range(0, n_superlayers):
    for l in range(0, n_layers_per_superlayer):
        total_number_of_layers += 1
        total_number_of_cells += n_cell_superlayer0 + sl * n_cell_increment
        n_cell_per_layer[str(n_layers_per_superlayer * sl + l)] = n_cell_superlayer0 + sl * n_cell_increment
print("total_number_of_cells: ", total_number_of_cells)
print("total_number_of_layers: ", total_number_of_layers)
print("n_cell_per_layer: ", n_cell_per_layer)
max_n_cell_per_layer = n_cell_per_layer[str(total_number_of_layers - 1)]

list_n_cells_fired_mc = []
occupancies_per_layer = []

avg_occupancies_per_layer = []
occupancies_per_layer_profile = []
overall_occupancies = [] #basically percentage of fired cells

dict_cellID_nHits = {}
# total_number_of_hit_integrated_per_batch = 0
number_of_cell_with_multiple_hits = 0
DC_fired_cell_map = []
dict_layer_phiSet = {} # cross check the number of cell

np.save(dic_file_path, dic)


    
    
    
#loop over all the files
for i in range(0, len(list_overlay)):
    dic = np.load(dic_file_path, allow_pickle=True).item()  
    list_index, list_hit_path_length, list_pos_hit, list_superLayer, \
        list_layer, list_stereo, list_nphi, list_pos_z = [], [], [], [], [], [], [], []
    list_R, list_p, list_px, list_py, list_pz, list_gen_status, \
        list_pos_ver, list_par_photon, list_pdg, \
            list_hits_secondary, list_hits_mc_secondary = [], [], [], [], [], [], [], [], [], [], []
    seen, count_hits = [], []
    # list_list2key = []

    rootfile = list_overlay[i]
    print(f"Running over file: {rootfile}")
    reader = root_io.Reader(rootfile)
    metadata = reader.get("metadata")[0]
    if type == "combined":
        cellid_encoding = metadata.get_parameter("CDCHHits__CellIDEncoding")
    else:
        cellid_encoding = metadata.get_parameter("DCHCollection__CellIDEncoding")
    decoder = dd4hep.BitFieldCoder(cellid_encoding)
    
    
    
    
    
    for event in reader.get("events"):
        if type == "combined":
            dc_hits = event.get("CDCHHits")
        else:
            dc_hits = event.get("DCHCollection")
        dict_particle_n_fired_cell = {}
        dict_particle_fired_cell_id = {}
        
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
            
            cellID = dc_hit.getCellID()
            superlayer = decoder.get(cellID, "superlayer")
            layer = decoder.get(cellID, "layer")
            nphi = decoder.get(cellID, "nphi")
            # stereo = decoder.get(cellID, "stereo")
            list_superLayer.append(superlayer)
            list_layer.append(layer)
            list_nphi.append(nphi)
            # list_stereo.append(stereo)
            
             # define a unique layer index based on super layer and layer
            if layer >= n_layers_per_superlayer or superlayer >= n_superlayers:
                print("Error: layer or super layer index out of range")
                print(f"Layer: {layer} while max layer is {n_layers_per_superlayer - 1}. \
                    Superlayer: {superlayer} while max superlayer is {n_superlayers - 1}.")
            unique_layer_index = superlayer * n_layers_per_superlayer + layer
            # cross check the number of cell
            if not unique_layer_index in dict_layer_phiSet.keys():
                dict_layer_phiSet[unique_layer_index] = {nphi}
            else:
                dict_layer_phiSet[unique_layer_index].add(nphi)
            cellID_unique_identifier = "SL_" + str(superlayer)  + "_L_" + str(layer) + "_nphi_" + str(nphi) 
            
            # what is the occupancy?
            if not cellID_unique_identifier in dict_cellID_nHits.keys(): # the cell was not fired yet
                #list_occupancy_per_layer.Fill(unique_layer_index)
                occupancies_per_layer.append(unique_layer_index)
                dict_cellID_nHits[cellID_unique_identifier] = 1
            else:
                if(dict_cellID_nHits[cellID_unique_identifier] == 1):
                    number_of_cell_with_multiple_hits += 1
                dict_cellID_nHits[cellID_unique_identifier] += 1
            
            # deal with the number of cell fired per particle
            # print(f"cellID_unique_identifier: {cellID_unique_identifier}")
            # print(dict_particle_fired_cell_id)
            if index_mc not in dict_particle_n_fired_cell.keys(): # the particle was not seen yet
                dict_particle_n_fired_cell[index_mc] = 1
                dict_particle_fired_cell_id[index_mc] = [cellID_unique_identifier]
            else: # the particle already fired cells
                if not cellID_unique_identifier in dict_particle_fired_cell_id[index_mc]: # this cell was not yet fired by this particle
                    dict_particle_n_fired_cell[index_mc] += 1
                    dict_particle_fired_cell_id[index_mc].append(cellID_unique_identifier)
                      
        # Where do the particles hit the DC?
        # DC_simhit_position_rz.Fill(abs(dc_hit.getPosition().z), sqrt(dc_hit.getPosition().x ** 2 + dc_hit.getPosition().y ** 2))
        # DC_simhit_position_xy.Fill(dc_hit.getPosition().x, dc_hit.getPosition().y)
        # # Map of the fired cells energies
        # DC_fired_cell_map.Fill(nphi, unique_layer_index, 1e+3*dc_hit.getEDep())
        # DC_fired_cell_map_per_evt.Fill(nphi, unique_layer_index, 1e+3*dc_hit.getEDep())
    
    #print(f"dict_particle_n_fired_cell: {dict_particle_n_fired_cell}")
    for particleKey in dict_particle_n_fired_cell.keys():
        # n_cell_fired_of_particles_hitting_dch.Fill(dict_particle_n_fired_cell[particleKey])
        # n_cell_fired_of_particles_hitting_dch_log.Fill(dict_particle_n_fired_cell[particleKey])
        list_n_cells_fired_mc.append(dict_particle_n_fired_cell[particleKey])
                
                
                
                
                

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
    percentage_of_fired_cells = 100 * len(dict_cellID_nHits.keys())/float(total_number_of_cells)       
    
    # Normalize the occupancy per layer th1 (divide the number of cell fired by the total number of cell) and fill the TProfile of occupancies
    for unique_layer_index in range(0, total_number_of_layers):
        print(f"unique_layer_index: {unique_layer_index}")
        # raw_bin_content = occupancies_per_layer.GetBinContent(unique_layer_index + 1) # NB: we use the trick that the bin index here is the same as the layer index it corresponds to, just binIdx 0 is underflow
        # occupancies_per_layer.SetBinContent(unique_layer_index + 1, 100 * raw_bin_content/float(n_cell_per_layer[str(unique_layer_index)])) # unique_layer_index and n_cell_per_layer key definitions coincide
        
        #basicaly, we are calculating the occupancy of each layer
        #so for each layer, we get the number of cells that were fired and divide by the total number of cells in that layer
            # avg_occupancies_per_layer.append(100 * len([x for x in occupancies_per_layer if x == unique_layer_index])/float(n_cell_per_layer[str(unique_layer_index)]))
        # Step 1: Filter occupancies_per_layer for the current layer index
        print(f"occupancies_per_layer: {occupancies_per_layer}")
        filtered_occupancies = [x for x in occupancies_per_layer if x == unique_layer_index]
        print(f"filtered_occupancies: {filtered_occupancies}")
        # Step 2: Count the occurrences of unique_layer_index
        layer_count = len(filtered_occupancies)
        print(f"layer_count: {layer_count}")
        # Step 3: Get the total number of cells for the layer
        total_cells_in_layer = float(n_cell_per_layer[str(unique_layer_index)])
        print(f"total_cells_in_layer: {total_cells_in_layer}")
        # Step 4: Calculate the percentage occupancy
        percentage_occupancy = 100 * layer_count / total_cells_in_layer
        # Step 5: Append the result to avg_occupancies_per_layer
        avg_occupancies_per_layer.append(percentage_occupancy)
        #profile adds a point where x is the layer index and y is the normalized occupancy
        occupancies_per_layer_profile.append( (unique_layer_index, percentage_occupancy) )
        #input("Press Enter to continue...")
    # overall_occupancies.append(percentage_of_fired_cells)
    
    
    #update dictionary
    dic["R"] = np.append(dic["R"], np.array(list_R))
    dic["p"] = np.append(dic["p"], np.array(list_p))
    dic["px"] = np.append(dic["px"], np.array(list_px))
    dic["py"] = np.append(dic["py"], np.array(list_py))
    dic["pz"] = np.append(dic["pz"], np.array(list_pz))
    dic["gens"] = np.append(dic["gens"], np.array(list_gen_status))
    dic["pos_ver"] = np.append(dic["pos_ver"], np.array(list_pos_ver))
    
    dic["hits"] = np.append(dic["hits"], np.array(list_index))
    dic["pos_hit"] = np.append(dic["pos_hit"], np.array(list_pos_hit))
    dic["unique_mcs"] = np.append(dic["unique_mcs"], np.array(unique_mcs))
    dic["superLayer"] = np.append(dic["superLayer"], np.array(list_superLayer))
    dic["layer"] = np.append(dic["layer"], np.array(list_layer))
    dic["nphi"] = np.append(dic["nphi"], np.array(list_nphi))
    dic["stereo"] = np.append(dic["stereo"], np.array(list_stereo))
    dic["pos_z"] = np.append(dic["pos_z"], np.array(list_pos_z))
    dic["count_hits"] = np.append(dic["count_hits"], np.array(count_hits))
    dic["has_par_photon"] = np.append(dic["has_par_photon"], np.array(list_par_photon))
    dic["pdg"] = np.append(dic["pdg"], np.array(list_pdg))
    dic["hits_produced_secondary"] = np.append(dic["hits_produced_secondary"], np.array(list_hits_secondary))
    dic["hits_mc_produced_secondary"] = np.append(dic["hits_mc_produced_secondary"], np.array(list_hits_mc_secondary))
    
    dic["percentage_of_fired_cells"] =  np.append(dic["percentage_of_fired_cells"], percentage_of_fired_cells)
    dic["list_n_cells_fired_mc"] = np.append(dic["list_n_cells_fired_mc"], np.array(list_n_cells_fired_mc))
    dic["occupancies_per_layer"] = np.append(dic["occupancies_per_layer"], np.array(occupancies_per_layer))
    dic["avg_occupancy"] = np.append(dic["avg_occupancy"], np.array(avg_occupancies_per_layer))
    dic["occupancies_per_layer_profile"] = np.append(dic["occupancies_per_layer_profile"], np.array(occupancies_per_layer_profile))
    dic["dict_layer_phiSet"] = dict_layer_phiSet
    
    dic["n_cell_per_layer"] = n_cell_per_layer
    dic["total_number_of_cells"] = total_number_of_cells
    dic["total_number_of_layers"] = total_number_of_layers
    dic["max_n_cell_per_layer"] = max_n_cell_per_layer
    

        
    np.save(dic_file_path, dic)