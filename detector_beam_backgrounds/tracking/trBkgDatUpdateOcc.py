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
if type == "combined":
    print("WARNING: combined files are not yet supported")
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

occ_keys = ["list_n_cells_fired_mc", "max_n_cell_per_layer",
        "n_cell_per_layer", "total_number_of_cells", "total_number_of_layers", 
        "occupancy_per_batch_sum_events", "occupancy_per_batch_sum_events_error",
        "occupancy_per_batch_sum_events_non_normalized", "occupancy_per_batch_sum_events_non_normalized_error"
        "occupancy_per_batch_sum_batches", "occupancy_per_batch_sum_batches_error",]
#check if dic_file_path exists:
try:
    dic = np.load(dic_file_path, allow_pickle=True).item() 
    print(f"Dictionary loaded from {dic_file_path}")
    for key in occ_keys:
        dic[key] = []
except:
    print(f"Dictionary not found at {dic_file_path}")
    print("Creating new dictionary")
    #assign dic to empty dictionary
    dic = {}
    for key in occ_keys:
        dic[key] = []
    np.save(dic_file_path, dic)
    # dic = np.load(dic_file_path, allow_pickle=True).item() 



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
if type=="bkg": #we want to get the occupancy for 20 events/files at a time
    batches=20
    eventFactor=1
elif type=="signal": #we want to get the occupancy for 1 event at a time
    batches=1
    eventFactor=10

occupancies_per_batch_sum_events = np.zeros((int(numfiles/batches), total_number_of_layers)) #we want it to be (500/20, 14) so 14 across 25 down
occupancies_per_batch_sum_events_non_normalized = np.zeros((int(numfiles/batches), total_number_of_layers))
occupancy_one_file = np.zeros(total_number_of_layers)
occupancies_non_normalized = np.zeros(total_number_of_layers)

#separate getting occupancy until at once per batch
occupancies_per_batch_sum_batch = np.zeros((int(eventFactor*numfiles/batches), total_number_of_layers)) #we want it to be (500/20, 14) so 14 across 25 down
occupancies_per_batch_sum_batch_non_normalized = np.zeros((int(eventFactor*numfiles/batches), total_number_of_layers))



numBatches = 0 #total batches should be numFiles / 20

#loop over all the files
for i in range(0, len(list_overlay)): 
    print(f"i: {i}")
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
        occupancies_a_batch_sum_each_event = np.zeros(total_number_of_layers)
        occupancies_a_batch_sum_each_event_non_normalized = np.zeros(total_number_of_layers)
        if type == "bkg": #want to reset every 20 bkg files
            # numBatches = 0 #total batches should be numFiles / 20
            print(f"resetting occupancy for batch, new batch: {numBatches}")
            dict_cellID_nHits = {} #reset cells for every 20 bkg event
            occupancies_a_batch = []
    
    
    numEvents = 0
    for event in reader.get("events"):
        numEvents += 1
        occupancies_an_event = []
        
        if type == "signal": #want to reset every 1 signal event
            dict_cellID_nHits = {} #reset cells for every 1 signal event
            occupancies_a_batch = []
            # numBatches = 0 #total batches should be numFiles * 10
    
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
            
            cellID = dc_hit.getCellID()
            superlayer = decoder.get(cellID, "superlayer")
            layer = decoder.get(cellID, "layer")
            nphi = decoder.get(cellID, "nphi")
            
             # define a unique layer index based on super layer and layer
            if layer >= n_layers_per_superlayer or superlayer >= n_superlayers:
                print("Error: layer or super layer index out of range")
                print(f"Layer: {layer} while max layer is {n_layers_per_superlayer - 1}. \
                    Superlayer: {superlayer} while max superlayer is {n_superlayers - 1}.")
            unique_layer_index = superlayer * n_layers_per_superlayer + layer
            cellID_unique_identifier = "SL_" + str(superlayer)  + "_L_" + str(layer) + "_nphi_" + str(nphi) 
            
            # what is the occupancy?
            if not cellID_unique_identifier in dict_cellID_nHits.keys(): # the cell was not fired yet
                occupancies_an_event.append(unique_layer_index)
                occupancies_a_batch.append(unique_layer_index)
                dict_cellID_nHits[cellID_unique_identifier] = 1
            else: # the cell was already fired
                dict_cellID_nHits[cellID_unique_identifier] += 1
            
            # deal with the number of cell fired per particle
            if index_mc not in dict_particle_n_fired_cell.keys(): # the particle was not seen yet
                dict_particle_n_fired_cell[index_mc] = 1
                dict_particle_fired_cell_id[index_mc] = [cellID_unique_identifier]
            else: # the particle already fired cells
                if not cellID_unique_identifier in dict_particle_fired_cell_id[index_mc]: # this cell was not yet fired by this particle
                    dict_particle_n_fired_cell[index_mc] += 1
                    dict_particle_fired_cell_id[index_mc].append(cellID_unique_identifier)
        #end of hit loop
        
        #calculate the occupancy for one event (so hits have accumulated over one event)
        event_occupancy = []
        event_occupancy_non_normalized = []
        for unique_layer_index in range(0, total_number_of_layers): #get the occupancy for a given layer 
            event_occupancy.append(calculateOccupancy(occupancies_an_event, unique_layer_index, n_cell_per_layer))
            #this gets the percent occupancy for a given layer for this event
            event_occupancy_non_normalized.append(calculateOccupancyNonNormalized(occupancies_an_event, unique_layer_index, n_cell_per_layer))

        occupancies_a_batch_sum_each_event += np.array(event_occupancy)
        occupancies_a_batch_sum_each_event_non_normalized += np.array(event_occupancy_non_normalized)
        
        ##signal files we want to reset every event
        if type == "signal":
            batch_occupancy = []
            for unique_layer_index in range(0, total_number_of_layers):
                batch_occupancy.append(calculateOccupancy(occupancies_a_batch, unique_layer_index, n_cell_per_layer))
            occupancies_per_batch_sum_batch[numBatches] = batch_occupancy #note index should be fileNum + eventNum / batches(1)
            numBatches += 1
        
        
    #the next one resets the 20 file batch
    if (i + 1) % batches == 0 and type == "bkg":
        print(f"setting occupancy for batch: {numBatches}")
        batch_occupancy = []
        for unique_layer_index in range(0, total_number_of_layers): #get the occupancy for a given layer 
            batch_occupancy.append(calculateOccupancy(occupancies_a_batch, unique_layer_index, n_cell_per_layer))
        occupancies_per_batch_sum_batch[numBatches] = batch_occupancy
        # occupancies_per_batch_sum_batch_non_normalized[numBatches] = occupancies_a_batch_sum_each_event_non_normalized
        numBatches += 1
        
        #old sum events
        occupancies_per_batch_sum_events[int((i+1)/batches) - 1] = occupancies_a_batch_sum_each_event
        occupancies_per_batch_sum_events_non_normalized[int((i+1)/batches) - 1] = occupancies_a_batch_sum_each_event_non_normalized
        
    
    
    
    for particleKey in dict_particle_n_fired_cell.keys():
        list_n_cells_fired_mc.append(dict_particle_n_fired_cell[particleKey])
                
                
    # percentage_of_fired_cells.append(100 * len(dict_cellID_nHits.keys())/float(total_number_of_cells)  )
#end of file loop

print("end of file loop")


print("updating dictionary")
#at this point we have filled the occupancy so there are 500 rows and 112 columns
dic["occupancy_per_batch_sum_events_non_normalized"] = np.mean(occupancies_per_batch_sum_events_non_normalized, axis=0)
dic["occupancy_per_batch_sum_events_non_normalized_error"] = np.std(occupancies_per_batch_sum_events_non_normalized, axis=0)
dic["occupancy_per_batch_sum_events"] = np.mean(occupancies_per_batch_sum_events, axis=0)
dic["occupancy_per_batch_sum_events_error"] = np.std(occupancies_per_batch_sum_events, axis=0)

dic["occupancy_per_batch_sum_batches"] = np.mean(occupancies_per_batch_sum_batch, axis=0)
dic["occupancy_per_batch_sum_batches_error"] = np.std(occupancies_per_batch_sum_batch, axis=0)

print(f"shape of occupancy_per_batch_sum_batches: {dic['occupancy_per_batch_sum_batches'].shape}")

# dic["percentage_of_fired_cells"] += percentage_of_fired_cells
dic["list_n_cells_fired_mc"] += list_n_cells_fired_mc
dic["n_cell_per_layer"] = n_cell_per_layer
dic["total_number_of_cells"] = total_number_of_cells
dic["total_number_of_layers"] = total_number_of_layers
dic["max_n_cell_per_layer"] = max_n_cell_per_layer


print(f"Saving dictionary to {dic_file_path}")
np.save(dic_file_path, dic)
