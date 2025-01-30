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
type = "signal"
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
        "hits_produced_secondary", "hits_mc_produced_secondary", 
        "list_n_cells_fired_mc", "dict_layer_phiSet", "max_n_cell_per_layer",
        "n_cell_per_layer", "total_number_of_cells", "total_number_of_layers", 
        "occupancy_one_file", "occupancy_one_file_non_normalized",
        "occupancy_per_batch_file", "occupancy_per_batch_file_error",
        "occupancy_per_batch_file_non_normalized", "occupancy_per_batch_file_non_normalized_error"]
#assign dic to empty dictionary
dic = {}
for key in keys:
    dic[key] = []
dic = np.load(dic_file_path, allow_pickle=True).item() 



def calculateOccupancy(occupancy, unique_layer_index, n_cell_per_layer):
    #basicaly, we are calculating the occupancy of each layer
    #so for each layer, we get the number of cells that were fired and divide by the total number of cells in that layer
    filtered_occupancies = [x for x in occupancy if x == unique_layer_index]
    layer_count = len(filtered_occupancies)
    total_cells_in_layer = float(n_cell_per_layer[str(unique_layer_index)])
    percentage_occupancy = 100 * layer_count / total_cells_in_layer
    return percentage_occupancy

def calculateOccupancyNonNormalized(occupancy, unique_layer_index, n_cell_per_layer):
    filtered_occupancies = [x for x in occupancy if x == unique_layer_index]
    layer_count = len(filtered_occupancies)
    return layer_count

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
number_of_cell_with_multiple_hits = 0
DC_fired_cell_map = []
dict_layer_phiSet = {} # cross check the number of cell






if type=="bkg":
    batches=20
elif type=="signal":
    batches=1

occupancies_per_batch_file = np.zeros((int(numfiles/batches), total_number_of_layers)) #we want it to be (500/20, 14) so 14 across 25 down
occupancies_per_batch_file_non_normalized = np.zeros((int(numfiles/batches), total_number_of_layers))
occupancy_one_file = np.zeros(total_number_of_layers)
occupancies_non_normalized = np.zeros(total_number_of_layers)


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
    
    #reset the batch file mean after starting a new batch
    if i % batches == 0:
        occupancies_a_batch_file = np.zeros(total_number_of_layers)
        occupancies_a_batch_file_non_normalized = np.zeros(total_number_of_layers)
        if type == "bkg": #want to reset every 20 bkg files
            dict_cellID_nHits = {} #reset cells for every 20 bkg event
    
    occupancies_a_file = []
    
    
    
    
    
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
        
        occupancies_an_event = []
        
        if type == "signal": #want to reset every 1 signal event
            dict_cellID_nHits = {} #reset cells for every 1 signal event
    
        if type == "":
            dc_hits = event.get("CDCHHits")
        else:
            dc_hits = event.get("DCHCollection")
        dict_particle_n_fired_cell = {}
        dict_particle_fired_cell_id = {}
        
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
                # occupancies_per_layer.append(unique_layer_index)
                occupancies_an_event.append(unique_layer_index)
                dict_cellID_nHits[cellID_unique_identifier] = 1
            else:
                if(dict_cellID_nHits[cellID_unique_identifier] == 1):
                    number_of_cell_with_multiple_hits += 1
                dict_cellID_nHits[cellID_unique_identifier] += 1
            
            # deal with the number of cell fired per particle
            if index_mc not in dict_particle_n_fired_cell.keys(): # the particle was not seen yet
                dict_particle_n_fired_cell[index_mc] = 1
                dict_particle_fired_cell_id[index_mc] = [cellID_unique_identifier]
            else: # the particle already fired cells
                if not cellID_unique_identifier in dict_particle_fired_cell_id[index_mc]: # this cell was not yet fired by this particle
                    dict_particle_n_fired_cell[index_mc] += 1
                    dict_particle_fired_cell_id[index_mc].append(cellID_unique_identifier)
            # overall_occupancies += (occupancies_an_event)
            occupancies_a_file += occupancies_an_event
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
        
        #calculate the occupancy for one event (so hits have accumulated over one event)
        event_occupancy = []
        event_occupancy_non_normalized = []
        for unique_layer_index in range(0, total_number_of_layers): #get the occupancy for a given layer 
            event_occupancy.append(calculateOccupancy(occupancies_an_event, unique_layer_index, n_cell_per_layer))
            event_occupancy_non_normalized.append(calculateOccupancyNonNormalized(occupancies_an_event, unique_layer_index, n_cell_per_layer))

        occupancies_a_batch_file += np.array(event_occupancy)
        occupancies_a_batch_file_non_normalized += np.array(event_occupancy_non_normalized)


        if i == 0:
            # print(f"event_occupancy: {event_occupancy}")
            occupancy_one_file = event_occupancy
            occupancy_one_file_non_normalized = event_occupancy_non_normalized
            dic["occupancy_one_file"] = occupancy_one_file
            dic["occupancy_one_file_non_normalized"] = occupancy_one_file_non_normalized

        
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
        
        # dic["avg_occupancy_event"] += [mean_occupancy_one_event]
        # print(f"avg_occupancy_event: {dic['avg_occupancy_event']}")
    #end of event loop
    
    #calculate the occupancy for one file (so hits have accumulated over all events)
    # file_occupancy = []
    # file_occupancy_non_normalized = []
    # for unique_layer_index in range(0, total_number_of_layers): #get the occupancy for a given layer 
    #     file_occupancy.append(calculateOccupancy(occupancies_an_event, unique_layer_index, n_cell_per_layer))
    #     file_occupancy_non_normalized.append(calculateOccupancyNonNormalized(occupancies_an_event, unique_layer_index, n_cell_per_layer))
    
    #the next one resets the 20 file batch
    if (i + 1) % batches == 0:
        # print(f"shape: {occupancies_per_batch_file.shape}")
        occupancies_per_batch_file[int((i+1)/batches) - 1] = occupancies_a_batch_file
        occupancies_per_batch_file_non_normalized[int((i+1)/batches) - 1] = occupancies_a_batch_file_non_normalized
        # print(f"occupancies_per_batch_file: {occupancies_per_batch_file}")
    
    
    for particleKey in dict_particle_n_fired_cell.keys():
        list_n_cells_fired_mc.append(dict_particle_n_fired_cell[particleKey])
                
                
    # percentage_of_fired_cells.append(100 * len(dict_cellID_nHits.keys())/float(total_number_of_cells)  )
#end of file loop

print("end of file loop")

#at this point we have filled the occupancy so there are 500 rows and 112 columns
dic["occupancy_per_batch_file_non_normalized"] = np.mean(occupancies_per_batch_file_non_normalized, axis=0)
dic["occupancy_per_batch_file_non_normalized_error"] = np.std(occupancies_per_batch_file_non_normalized, axis=0)
dic["occupancy_per_batch_file"] = np.mean(occupancies_per_batch_file, axis=0)
dic["occupancy_per_batch_file_error"] = np.std(occupancies_per_batch_file, axis=0)

# dic["percentage_of_fired_cells"] += percentage_of_fired_cells
dic["list_n_cells_fired_mc"] += list_n_cells_fired_mc
dic["dict_layer_phiSet"] += dict_layer_phiSet
dic["n_cell_per_layer"] = n_cell_per_layer
dic["total_number_of_cells"] = total_number_of_cells
dic["total_number_of_layers"] = total_number_of_layers
dic["max_n_cell_per_layer"] = max_n_cell_per_layer


print(f"Saving dictionary to {dic_file_path}")
np.save(dic_file_path, dic)
