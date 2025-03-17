#Alexander Poulin Jan 2025
from podio import root_io
import numpy as np 
import math
import dd4hep as dd4hepModule
from ROOT import dd4hep
import sys
from trBkgDat import configure_paths, setUpFiles
import argparse
# from multiKdOcc import calculateOnlyNeighborsKDTree
from scipy.spatial import KDTree
import time

"""
This script is used to update the occupancy of the background particles in the dictionary.
This can be run standalone just make sure to update filePaths accordingly.
This is meant to be ran after trBkgDat.py (which save a .npy)
but doesnt have to, it will just create a new .npy.
"""

# NoNeighborsRemoved = 0
# NeighborsRemained = 0

# EdepNeighborsRemained = 0
# NoEdepNeighborsRemoved = 0

def calculateOccupancy(occupancy, unique_layer_index, n_cell_per_layer):
    #basicaly, we are calculating the occupancy of each layer
    #so for each layer, we get the number of cells that were fired and divide by the total number of cells in that layer
    filtered_occupancies = [x for x in occupancy if x == unique_layer_index]
    layer_count = len(filtered_occupancies)
    total_cells_in_layer = float(n_cell_per_layer[str(unique_layer_index)])
    percentage_occupancy = 100 * layer_count / total_cells_in_layer
    return percentage_occupancy

def calculateOccupancyNonNormalized(occupancy, unique_layer_index, n_cell_per_layer):
    #calculate occupancy but just number of fired cells per layer pretty much
    filtered_occupancies = [x for x in occupancy if x == unique_layer_index]
    layer_count = len(filtered_occupancies)
    return layer_count

def calculateOnlyNeighbors(occupancy :list[tuple], edep :list[tuple], radiusR=1, radiusPhi=-1, atLeast=1, maxnphi=180, edepRange=0.05, edepAtLeast=1, 
                           NoNeighborsRemoved=0, NeighborsRemained=0, EdepNeighborsRemained=0, NoEdepNeighborsRemoved=0, maxLayer=112, 
                           edepLoosen=False, n_cell_per_layer=[]):
    #calculate the occupancy of non-neighbor cells
    #we will loop over all the cells and check if they have neighbors
    #a neightbor will be defined if there exists an occupancy index where (unique_layer_index +-0 or 1, nphi +- 0 or 1) exists
    #if they do, we will remove them from the list
    #occupancy is a list of tuples (unique_layer_index, nphi)
    #we will return a list of unique_layer_index
    dicNeighbors = {} #will be a dictionary where key is pos of some cell fired, the value will be a list of neighbor pos
    dicEdepNeighbors = {} #setup dictionary for current cell's neighbors edep
    
    only_neighbors = []
    only_neighbors_pos = []
    
    only_neighbors_edep = [] #edep of only neighbors
    only_neighbors_only_edep = [] #occ of only neighbors within range of edep
    only_neighbors_only_edep_edep = [] #edep of only neighbors within range of edep
    only_neighbors_only_edeps_pos = []
    
    #turn occupancy into a set (allow for faster lookup)
    setOccupancy = {}
    for i in range(0, len(occupancy)):
        setOccupancy[occupancy[i]] = True
    # print(f"setOccupancy: {setOccupancy}")
    
    dicEdep = {}
    #turn edep into a dictionary
    for i in range(0, len(edep)):
        key = (edep[i][0], edep[i][1])
        if key not in dicEdep.keys():
            dicEdep[key] = edep[i][2]
        else:
            dicEdep[key] += edep[i][2]
    # print(f"dicEdep: {dicEdep}")
    # input("Press Enter to continue...")
    
    
    if radiusPhi == -1:
        radiusPhi = radiusR
    # print(f"calculateNNOcc: {np.array(occupancy)}")
    for i in range(0, len(occupancy)):
        unique_layer_index = occupancy[i][0]
        nphi = occupancy[i][1]
        currentEdep = dicEdep[(unique_layer_index, nphi)]

        # print(f"unique_layer_index: {unique_layer_index}, nphi: {nphi}, currentEdeppos: {edep[i][0]}, {edep[i][1]}, currentEdep: {currentEdep}")
        
        neighbors = False
        neighborsEdep = False
        numNeighbors = 0
        numEdepNeighbors = 0
        
        #setup current cell if not seen before
        if (unique_layer_index, nphi) not in dicNeighbors:
            dicNeighbors[(unique_layer_index, nphi)] = [] #setup dictionary for current cell's neighbors
        if (unique_layer_index, nphi) not in dicEdepNeighbors:
            dicEdepNeighbors[(unique_layer_index, nphi)] = [] #setup dictionary for current cell's neighbors edep
        

        
        for dx in range(-radiusR, radiusR + 1):
            # print(f"len(dicNeighbors): {len(dicNeighbors[(unique_layer_index, nphi)])} where atleast: {atLeast}")
            # print(f"len(dicEdepNeighbors): {len(dicEdepNeighbors[(unique_layer_index, nphi)])} where edepAtLeast: {edepAtLeast}")
            # input("Press Enter to continue...")
            if len(dicNeighbors[(unique_layer_index, nphi)]) >= atLeast and len(dicEdepNeighbors[(unique_layer_index, nphi)]) >= edepAtLeast: #have we already seen enough
                neighbors = True
                neighborsEdep = True
                # print("able to skip")
                break #skip if we have already seen enough neighbors and break out of dx loop
            
            for dy in range(-radiusPhi, radiusPhi + 1):
                if dx == 0 and dy == 0: # Skip the center point
                    continue
                if dx == 0:  #skip same layer
                    continue
                if unique_layer_index + dx < 0 or unique_layer_index + dx >= maxLayer: #check boundaries ###fixx
                    continue
                
                
                cyclic_nphi = (nphi + dy) % maxnphi  # Wrap around for cyclic nphi #we will assume 180 for now
                cyclic_unique_layer_index = unique_layer_index + dx
                if len(dicNeighbors[(unique_layer_index, nphi)]) < atLeast and (cyclic_unique_layer_index, cyclic_nphi) in setOccupancy and (cyclic_unique_layer_index, cyclic_nphi) not in dicNeighbors[(unique_layer_index, nphi)]: 
                    #not already over nieghbor atleast
                    #nieghbor exists (i.e. has been fired) (then assume also exists in edep)
                    #and it hasnt already been counted in dicNeighbors
                    numNeighbors += 1
                    
                    dicNeighbors[(unique_layer_index, nphi)].append((cyclic_unique_layer_index, cyclic_nphi)) #add neighbor to cell
                    if (cyclic_unique_layer_index, cyclic_nphi) not in dicNeighbors:
                        dicNeighbors[(cyclic_unique_layer_index, cyclic_nphi)] = []
                    dicNeighbors[(cyclic_unique_layer_index, cyclic_nphi)].append((unique_layer_index, nphi)) #add cell to neighbor (reduce double counting)
                    
                if len(dicEdepNeighbors[(unique_layer_index, nphi)]) < edepAtLeast and (cyclic_unique_layer_index, cyclic_nphi) in dicEdep and (cyclic_unique_layer_index, cyclic_nphi) not in dicEdepNeighbors[(unique_layer_index, nphi)]:
                    neighborEdep = dicEdep[(cyclic_unique_layer_index, cyclic_nphi)]
                    if abs(currentEdep - neighborEdep) <= edepRange: #if neighbor within range of edep
                        numEdepNeighbors += 1
                        dicEdepNeighbors[(unique_layer_index, nphi)].append((cyclic_unique_layer_index, cyclic_nphi)) #add neighbor to cell
                        if (cyclic_unique_layer_index, cyclic_nphi) not in dicEdepNeighbors:
                            dicEdepNeighbors[(cyclic_unique_layer_index, cyclic_nphi)] = []
                        dicEdepNeighbors[(cyclic_unique_layer_index, cyclic_nphi)].append((unique_layer_index, nphi)) #add cell to neighbor (reduce double counting)
                        
                if numNeighbors >= atLeast:
                    neighbors = True
                if numEdepNeighbors >= edepAtLeast:
                    neighborsEdep = True
                if neighborsEdep and neighbors: #this may be redundant due to first check
                    # print("break early")
                    break
            #end of dy loop
        #end of dx loop

        #determine outcome of cell:
        if neighbors: #if neighbors, add to only_neighbors
            only_neighbors.append(unique_layer_index)
            only_neighbors_pos.append((unique_layer_index, nphi))
            only_neighbors_edep.append((unique_layer_index, nphi, currentEdep)) #also should do edep for only neighbors only edep
            if type(NoNeighborsRemoved) == int:
                NeighborsRemained += 1
            else:
                NeighborsRemained.value += 1
        else:
            if type(NoNeighborsRemoved) == int:
                NoNeighborsRemoved += 1
            else:
                NoNeighborsRemoved.value += 1
        if neighborsEdep:
            only_neighbors_only_edeps_pos.append((unique_layer_index, nphi))
            only_neighbors_only_edep.append(unique_layer_index)
            only_neighbors_only_edep_edep.append((unique_layer_index, nphi, currentEdep))
            if type(EdepNeighborsRemained) == int:
                EdepNeighborsRemained += 1
            else:
                EdepNeighborsRemained.value += 1
        else:
            if type(NoEdepNeighborsRemoved) == int:
                NoEdepNeighborsRemoved += 1
            else:
                NoEdepNeighborsRemoved.value += 1    
    
    
    #currently not returning NoNeighborsRemoved so its just a copy right now, the final value will not be correct
    return only_neighbors, only_neighbors_pos, only_neighbors_only_edep, only_neighbors_edep, only_neighbors_only_edep_edep, only_neighbors_only_edeps_pos, \
        NoNeighborsRemoved, NeighborsRemained, EdepNeighborsRemained, NoEdepNeighborsRemoved


def build_kdtrees(list_of_A):
    """Builds a KDTree for each sublist in list_of_A."""
    return [KDTree(A) for A in list_of_A]

def calculateOnlyNeighborsKDTree(occupancy :list[tuple], edep :list[tuple], radiusR=1, radiusPhi=-1, atLeast=1, maxnphi=180, edepRange=0.05, edepAtLeast=1, 
                           NoNeighborsRemoved=0, NeighborsRemained=0, EdepNeighborsRemained=0, NoEdepNeighborsRemoved=0):
    #calculate the occupancy of non-neighbor cells
    #we will loop over all the cells and check if they have neighbors
    #a neightbor will be defined if there exists an occupancy index where (unique_layer_index +-0 or 1, nphi +- 0 or 1) exists
    #if they do, we will remove them from the list
    #occupancy is a list of tuples (unique_layer_index, nphi)
    #we will return a list of unique_layer_index
    # print("calculateOnlyNeighbors")
    only_neighbors = []
    only_neighbors_pos = []
    
    # print(edep)
    only_neighbors_edep = [] #edep of only neighbors
    only_neighbors_only_edep = [] #occ of only neighbors within range of edep
    only_neighbors_only_edep_edep = [] #edep of only neighbors within range of edep
    only_neighbors_only_edeps_pos = []
    # no_neighbors_removed_layers = []
    # neighbors_remained_layers = []
    if radiusPhi == -1:
        radiusPhi = radiusR
        
    coords = np.array(occupancy)
    kdTree = KDTree(coords)
    energies = np.array([b[2] for b in edep])
    
    print(f"calculateNNOcc: {np.array(occupancy)}")
    for i, (x, y) in enumerate(occupancy):
        neighbors = False
        neighborsEdep = False
        unique_layer_index, nphi = occupancy[i]
        # Find spatial neighbors within (x_range, y_range)
        neighbors_idx = kdTree.query_ball_point((x, y), r=max(radiusR, radiusPhi))  # Get candidates
        
        # Extract neighbor coordinates
        neighbor_coords = coords[neighbors_idx]
        
        neighbor_count = len(neighbors_idx)
        
        # Apply manual bounding box filtering (since KDTree uses a circular search)
        valid_spatial = [
            idx for idx, (nx, ny) in zip(neighbors_idx, neighbor_coords)
            if abs(nx - x) <= radiusR and abs(ny - y) <= radiusPhi
        ]
        
        # Energy filtering
        valid_neighbors = [
            idx for idx in valid_spatial
            if abs(energies[idx] - energies[i]) <= edepRange
        ]
        
        # Store count of valid neighbors
        edepNeighbor_count = len(valid_neighbors)
        
        if neighbor_count >= atLeast:
            neighbors == True
            only_neighbors.append(unique_layer_index)
            only_neighbors_pos.append((unique_layer_index, nphi))
            only_neighbors_edep.append((unique_layer_index, nphi, edep[i][2])) #also should do edep for only neighbors only edep
            # neighbors_remained_layers.append(unique_layer_index)
            # global NeighborsRemained
            if type(NoNeighborsRemoved) == int:
                NeighborsRemained += 1
            else:
                NeighborsRemained.value += 1
        else:
            if type(NoNeighborsRemoved) == int:
                NoNeighborsRemoved += 1
            else:
                NoNeighborsRemoved.value += 1
            
            
        if edepNeighbor_count >= edepAtLeast:
            neighborsEdep = True
            only_neighbors_only_edeps_pos.append((unique_layer_index, nphi))
            only_neighbors_only_edep.append(unique_layer_index)
            only_neighbors_only_edep_edep.append((unique_layer_index, nphi, edep[i][2]))
            # global EdepNeighborsRemained
            if type(EdepNeighborsRemained) == int:
                # print("int")
                EdepNeighborsRemained += 1
            else:
                # print("manager")
                EdepNeighborsRemained.value += 1
        else:
            # print("edepNoEdepNeighborsRemoved")
            # global NoEdepNeighborsRemoved
            if type(NoEdepNeighborsRemoved) == int:
                # print("int")
                NoEdepNeighborsRemoved += 1
            else:
                # print("manager")
                NoEdepNeighborsRemoved.value += 1
    
    
    #currently not returning NoNeighborsRemoved so its just a copy right now, the final value will not be correct
    return only_neighbors, only_neighbors_pos, only_neighbors_only_edep, only_neighbors_edep, only_neighbors_only_edep_edep, only_neighbors_only_edeps_pos, \
        NoNeighborsRemoved, NeighborsRemained, EdepNeighborsRemained, NoEdepNeighborsRemoved
        
        
def calcOcc(occupancies_a_batch, occupancies_a_batch_only_neighbor, occupancies_a_batch_edep, 
            dic_occupancies_per_batch_sum_batch_energy_dep_per_cell, 
            n_cell_per_layer, total_number_of_layers, radiusR, radiusPhi, atLeast, edepRange, edepAtLeast,
            max_n_cell_per_layer, cell_to_mcID, 
            NoNeighborsRemoved, NeighborsRemained, EdepNeighborsRemained, NoEdepNeighborsRemoved, edepLoosen, useKDTree=False):
    """Generates all necessary occupancy data for a batch of events

    Args:
        occupancies_a_batch (_type_): _description_
        occupancies_a_batch_only_neighbor (_type_): _description_
        occupancies_a_batch_edep (_type_): _description_
        occupancies_a_batch_edep_per_cell (_type_): _description_
        dic_occupancies_per_batch_sum_batch_energy_dep_per_cell (_type_): _description_
        n_cell_per_layer (_type_): _description_
        total_number_of_layers (_type_): _description_
        radiusR (_type_): _description_
        radiusPhi (_type_): _description_
        atLeast (_type_): _description_
        max_n_cell_per_layer (_type_): _description_

    Returns:
        _type_: _description_
            batch_occupancy: for occupancies_per_batch_sum_batch
            batch_occupancy_only_neighbor: for occupancies_per_batch_sum_batch_only_neighbor
            batch_occupancy_edep: for occupancies_per_batch_sum_batch_energy_dep
            dic_occupancies_per_batch_sum_batch_energy_dep_per_cell: for dic_occupancies_per_batch_sum_batch_energy_dep_per_cell
    """
    
    
    batch_occupancy = []
    for unique_layer_index in range(0, total_number_of_layers):
        batch_occupancy.append(calculateOccupancy(occupancies_a_batch, unique_layer_index, n_cell_per_layer))
    # occupancies_per_batch_sum_batch[numBatches] = batch_occupancy #note index should be fileNum + eventNum / batches(1)
    
    
    #now determine non-neighbor occupancy
    batch_occupancy_only_neighbor = []
    batch_occupancy_only_neighbor_only_edep = []
    if useKDTree:
        # print("Using KDTree")
        occupancies_a_batch_only_neighbor_remain, only_neighbor_pos, occupancies_a_batch_only_neighbor_only_edep, \
            edep_only_neighbors, edep_only_neighbors_only_edep, only_neighbor_only_edep_pos, \
                NoNeighborsRemoved, NeighborsRemained, EdepNeighborsRemained, NoEdepNeighborsRemoved = calculateOnlyNeighborsKDTree(occupancies_a_batch_only_neighbor, occupancies_a_batch_edep, 
                                                                                        radiusR=radiusR, radiusPhi=radiusPhi, atLeast=atLeast, 
                                                                                        maxnphi=max_n_cell_per_layer, edepRange=edepRange, edepAtLeast=edepAtLeast,
                                                                                        NoNeighborsRemoved=NoNeighborsRemoved, NeighborsRemained=NeighborsRemained,
                                                                                        EdepNeighborsRemained=EdepNeighborsRemained, NoEdepNeighborsRemoved=NoEdepNeighborsRemoved, 
                                                                                        edepLoosen=edepLoosen, n_cell_per_layer=n_cell_per_layer)
    else:
        # print("noKDTRee")
        occupancies_a_batch_only_neighbor_remain, only_neighbor_pos, occupancies_a_batch_only_neighbor_only_edep, \
            edep_only_neighbors, edep_only_neighbors_only_edep, only_neighbor_only_edep_pos, \
                NoNeighborsRemoved, NeighborsRemained, EdepNeighborsRemained, NoEdepNeighborsRemoved = calculateOnlyNeighbors(occupancies_a_batch_only_neighbor, occupancies_a_batch_edep,
                                                                                        radiusR=radiusR, radiusPhi=radiusPhi, atLeast=atLeast, 
                                                                                        maxnphi=max_n_cell_per_layer, edepRange=edepRange, edepAtLeast=edepAtLeast,
                                                                                        NoNeighborsRemoved=NoNeighborsRemoved, NeighborsRemained=NeighborsRemained,
                                                                                        EdepNeighborsRemained=EdepNeighborsRemained, NoEdepNeighborsRemoved=NoEdepNeighborsRemoved, 
                                                                                        edepLoosen=edepLoosen, n_cell_per_layer=n_cell_per_layer) 
    #for each batch, occupancies_a_batch_only_neighbor is a list of tuples but we will return a list of unique_layer_index
    for unique_layer_index in range(0, total_number_of_layers):
        batch_occupancy_only_neighbor.append(calculateOccupancy(occupancies_a_batch_only_neighbor_remain, unique_layer_index, n_cell_per_layer))
        batch_occupancy_only_neighbor_only_edep.append(calculateOccupancy(occupancies_a_batch_only_neighbor_only_edep, unique_layer_index, n_cell_per_layer))
    # occupancies_per_batch_sum_batch_only_neighbor[numBatches] = batch_occupancy_only_neighbor
    
    #for energy deposition, we want to append the sum of energy deposition for each layer
    batch_occupancy_edep = []
    for unique_layer_index in range(0, total_number_of_layers):
        filtered_occupancies = [x for x in occupancies_a_batch_edep if x[0] == unique_layer_index]
        layer_edep_sum = sum([x[2] for x in filtered_occupancies]) #sums all cell's edep in layer
        batch_occupancy_edep.append(layer_edep_sum) 
    # occupancies_per_batch_sum_batch_energy_dep[numBatches] = batch_occupancy_edep
    
    # we make the layer, nphi, edep tuple into a dictionary
    #for each entry we will have a tuple (unique_layer_index,nphi) as the key and the accumulated edep as the value
    for i in range(0, len(occupancies_a_batch_edep)):
        key = (occupancies_a_batch_edep[i][0], occupancies_a_batch_edep[i][1])
        if key not in dic_occupancies_per_batch_sum_batch_energy_dep_per_cell.keys():
            dic_occupancies_per_batch_sum_batch_energy_dep_per_cell[key] = np.array([occupancies_a_batch_edep[i][2]])
        else:
            #so we will append to the key, but we will mean this later on... in the end we want just a dictionary of key to single edep value
            dic_occupancies_per_batch_sum_batch_energy_dep_per_cell[key] = np.append(dic_occupancies_per_batch_sum_batch_energy_dep_per_cell[key], occupancies_a_batch_edep[i][2])
            #kinda weird, we take this in and then return it, need to double check if we can just mutate it or if this is a copy (i believe it is a copy)
            
    occupancies_a_batch_only_neighbor_difference = np.array(batch_occupancy) - np.array(batch_occupancy_only_neighbor)     
    
    #given only_neighbor_pos, we will now calculate the mcID of the cells that were fired
    cell_to_mcID_neighbors = []
    for i in range(0, len(only_neighbor_pos)):
        unique_layer_index = only_neighbor_pos[i][0]
        nphi = only_neighbor_pos[i][1]
        #find where in cell_to_mcID the unique_layer_index is
        index = [i for i, tup in enumerate(cell_to_mcID) if tup[0] == unique_layer_index and tup[1] == nphi]
        if len(index) == 0:
            print("Error: unique_layer_index not found in cell_to_mcID")
        else:
            cell_to_mcID_neighbors.append(cell_to_mcID[index[0]])   
            
    cell_to_mcID_neighbors_only_edep = []
    for i in range(0, len(only_neighbor_only_edep_pos)):
        unique_layer_index = only_neighbor_only_edep_pos[i][0]
        nphi = only_neighbor_only_edep_pos[i][1]
        #find where in cell_to_mcID the unique_layer_index is
        index = [i for i, tup in enumerate(cell_to_mcID) if tup[0] == unique_layer_index and tup[1] == nphi]
        if len(index) == 0:
            print("Error: unique_layer_index not found in cell_to_mcID")
        else:
            cell_to_mcID_neighbors_only_edep.append(cell_to_mcID[index[0]])
    
    # print(f"occupancies_a_batch_only_neighbor_only_edep: {occupancies_a_batch_only_neighbor_only_edep}")
    # print(f"edep_only_neighbors: {edep_only_neighbors}")
    return batch_occupancy, batch_occupancy_only_neighbor, batch_occupancy_edep, \
        dic_occupancies_per_batch_sum_batch_energy_dep_per_cell, only_neighbor_pos, cell_to_mcID_neighbors, \
            batch_occupancy_only_neighbor_only_edep, edep_only_neighbors, edep_only_neighbors_only_edep, cell_to_mcID_neighbors_only_edep, \
                NoNeighborsRemoved, NeighborsRemained, EdepNeighborsRemained, NoEdepNeighborsRemoved
    
def updateOcc(typeFile="bkg", numfiles=500, radiusR=1, radiusPhi=-1, atLeast=1, edepRange=0.05, edepAtLeast=1, edepLoosen=False, flexible=True):
    print("Calculating occupancy data from files...")
    list_overlay = []
    # numfiles = 500
    # typeFile = "bkg"
    # flexible = True ##basically allows the list_overlay to skip over files that dont work
    if radiusPhi == -1:
        radiusPhi = radiusR
    # print(edepRange)
    #setup dictionary
    dic = {}
    #can change dic_file_path to the correct path:
    dic_file_path = "/eos/user/a/alpoulin/fccBBTrackData/noOcc/" + str(typeFile) + "_background_particles_" + str(numfiles) + ".npy" #cernbox (to save storage)
    output_dic_file_path = "/eos/user/a/alpoulin/fccBBTrackData/wEdep/" + str(typeFile) + "_background_particles_" + str(numfiles)  + "_v6" + \
        "_R" + str(radiusR) + "_P" + str(radiusPhi) + "_AL" + str(atLeast) + "_ER" + str(edepRange) + "_EAL" + str(edepAtLeast) + ".npy" #cernbox (to save storage)
    occ_keys = ["list_n_cells_fired_mc", "max_n_cell_per_layer",
        "n_cell_per_layer", "total_number_of_cells", "total_number_of_layers", 
        "occupancy_per_batch_sum_batch_non_normalized", "occupancy_per_batch_sum_batch_non_normalized_error"
        "occupancy_per_batch_sum_batches", "occupancy_per_batch_sum_batches_error", "occupancy_per_batch_sum_batches_non_meaned",
        "occupancy_per_batch_sum_batches_only_neighbor", "occupancy_per_batch_sum_batches_only_neighbor_error",
        "occupancy_per_batch_sum_batches_only_neighbor_only_edep", "occupancy_per_batch_sum_batches_only_neighbor_only_edep_error",
        "dic_occupancy_per_batch_sum_batches_energy_dep", "energy_dep_per_cell_per_batch",
        "occupancy_per_batch_sum_batch_avg_energy_dep", "occupancy_per_batch_sum_batch_avg_energy_dep_error",
        "combined_onlyBkg_occupancy_per_batch_sum_batches", "combined_onlyBkg_occupancy_per_batch_sum_batches_error",
        "combined_onlyBkg_occupancy_per_batch_sum_batches_only_neighbor", "combined_onlyBkg_occupancy_per_batch_sum_batches_only_neighbor_error",
        "combined_onlySignal_occupancy_per_batch_sum_batches", "combined_onlySignal_occupancy_per_batch_sum_batches_error",
        "combined_onlySignal_occupancy_per_batch_sum_batches_only_neighbor", "combined_onlySignal_occupancy_per_batch_sum_batches_only_neighbor_error",
        "no_neighbors_removed", "neighbors_remained", "occupancy_only_neighbor_difference", "occupancy_only_neighbor_difference_error",
        "cell_fired_pos", "cell_fired_pos_neighbors", "cell_fired_pos_by_batch","cellFiredMCID_per_batch", "onlyNeighborMCID_per_batch", 
        "neighborPt_by_batch", "neighborPDG_by_batch"]
    #check if dic_file_path exists:
    try:
        dic = np.load(dic_file_path, allow_pickle=True).item()
        print(f"Dictionary loaded from {dic_file_path}")
        for key in occ_keys:
            dic[key] = []
    except:
        print(f"Dictionary not found at {dic_file_path}")
        input("Press Enter to continue...")
        print("Creating new dictionary")
        #assign dic to empty dictionary
        dic = {}
        for key in occ_keys:
            dic[key] = []
        np.save(dic_file_path, dic)
        

    # print(f"typeFile: {typeFile}")
    print(f"output_dic_file_path once finished will be: {output_dic_file_path}")
    
    bkgDataPath, combinedDataPath, bkgFilePath, combinedFilePath, signalFilePath, signalDataPath = configure_paths(typeFile)

    list_overlay = setUpFiles(typeFile, flexible, numfiles, bkgDataPath, combinedDataPath, bkgFilePath, combinedFilePath, signalFilePath, signalDataPath)


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
    if typeFile=="bkg": #we want to get the occupancy for 20 events/files at a time
        batches=20
        eventFactor=1 #one event per file
    elif typeFile=="signal": #we want to get the occupancy for 1 event at a time
        batches=1
        eventFactor=10 #10 events per file
    elif typeFile=="combined": #we want to get the occupancy for 1 event at a time
        batches=1
        eventFactor=10 #10 events per file

    #separate getting occupancy until at once per batch
    occupancies_per_batch_sum_batch = np.zeros((int(eventFactor*numfiles/batches), total_number_of_layers)) #we want it to be (500/20, 14) so 14 across 25 down
    occupancies_per_batch_sum_batch_non_normalized = np.zeros((int(eventFactor*numfiles/batches), total_number_of_layers))
    
    occupancies_per_batch_sum_batch_only_bkg = np.zeros((int(eventFactor*numfiles/batches), total_number_of_layers)) #only used for combined
    occupancies_per_batch_sum_batch_only_signal = np.zeros((int(eventFactor*numfiles/batches), total_number_of_layers)) #only used for combined

    occupancies_per_batch_sum_batch_only_neighbor = np.zeros((int(eventFactor*numfiles/batches), total_number_of_layers))
    occupancies_per_batch_sum_batch_only_neighbors_only_edeps = np.zeros((int(eventFactor*numfiles/batches), total_number_of_layers))
    
    occupancies_per_batch_sum_batch_energy_dep = np.zeros((int(eventFactor*numfiles/batches), total_number_of_layers))
    dic_occupancies_per_batch_sum_batch_energy_dep_per_cell = {} #this will be a dictionary of np arrays
    energy_dep_per_cell_per_batch = []
    energy_dep_per_cell_per_batch_only_neighbors = []
    energy_dep_per_cell_per_batch_only_neighbors_only_edeps = []
    
    cell_fired_pos_per_batch = [] #should be appending such that it matches the total number of bathces
    cell_fired_pos_neighbors_per_batch = []
    
    
    cellFiredPos = []
    cellFiredPosNeighbors = []
    
    cell_to_mcID_per_batch = [] #a list of tuples (unique_layer_index, nphi, mcID) for each cell fired for each batch
    cell_to_mcID_neighbors_per_batch = [] #a list of tuples (unique_layer_index, nphi, mcID) for each cell fired for each batch
    cell_to_mcID_neighbors_edeps_per_batch = [] #a list of tuples (unique_layer_index, nphi, mcID) for each cell fired for each batch
    all_mcID_per_batch = [] #a list of all mcID's which fired cells for each batch
    
    neighborPt = []  #will be a list of lists of tuple of (radiusR, radiusPhi, pt)
    neighborPDG = [] #list of lists of tuple of (radiusR, radiusPhi, pdg)
    
    lowPt = [] #will be a list of lists of tuple of (radiusR, radiusPhi, pt)
    highPt = [] #will be a list of lists of tuple of (radiusR, radiusPhi, pt)
    
    NoNeighborsRemoved = 0
    NeighborsRemained = 0
    EdepNeighborsRemained = 0
    NoEdepNeighborsRemoved = 0


    numBatches = 0 
    #total batches for bkg should be numFiles / 20
    #total batches for signal should be numFiles * 10

    #loop over all the files
    for i in range(0, len(list_overlay)): 
        # print(f"i: {i}")
        rootfile = list_overlay[i]
        print(f"Running over file: {rootfile}")
        reader = root_io.Reader(rootfile)
        metadata = reader.get("metadata")[0]
        if typeFile == "":
            cellid_encoding = metadata.get_parameter("CDCHHits__CellIDEncoding")
        else:
            cellid_encoding = metadata.get_parameter("DCHCollection__CellIDEncoding")
        decoder = dd4hep.BitFieldCoder(cellid_encoding)
        # print("fileread")
        
        
        
        #reset the batch file mean after starting a new batch
        if i % batches == 0:
            # occupancies_a_batch_sum_each_event = np.zeros(total_number_of_layers)
            # occupancies_a_batch_sum_each_event_non_normalized = np.zeros(total_number_of_layers)
            if typeFile == "bkg": #want to reset every 20 bkg files
                # numBatches = 0 #total batches should be numFiles / 20
                print(f"resetting occupancy for batch, new batch: {numBatches}")
                dict_cellID_nHits = {} #reset cells for every 20 bkg event
                occupancies_a_batch = []
                occupancies_a_batch_only_neighbor = [] # a list of tuples (unique_layer_index, nphi)
                occupancies_a_batch_edep = [] # a list of tuples (unique_layer_index, edep)
                # occupancies_a_batch_edep_per_cell = [] #a list of tuples (unique_layer_index, nphi, edep)
                batch_cell_fired_pos_neighbors = []
                batch_cell_fired_pos = []
                cell_to_mcID = [] #a list of tuples (unique_layer_index, nphi, mcID) for each cell fired for each batch
                cell_to_mcID_neighbors = [] #a list of tuples (unique_layer_index, nphi, mcID) for each cell fired for each batch
                all_mcID = [] #a list of mcID's which fired cells for each batch
                batch_pt = [] #will be a list of tuple of (radiusR, radiusPhi, pt)
                batch_pdg = [] #will be a list of tuple of (radiusR, radiusPhi, pdg)
                batch_low_pt = [] #will be a list of tuple of (radiusR, radiusPhi, pt)
                batch_high_pt = [] #will be a list of tuple of (radiusR, radiusPhi, pt)
        
        
        numEvents = 0
        for event in reader.get("events"):
            # print(f"Running over event: {numEvents}")
            numEvents += 1
            occupancies_an_event = []
            
            EventMCParticles = event.get("MCParticles")
            
            if typeFile == "signal": #want to reset every 1 signal event
                print(f"resetting occupancy for batch, new batch: {numBatches}")
                dict_cellID_nHits = {} #reset cells for every 1 signal event
                occupancies_a_batch = []
                occupancies_a_batch_only_neighbor = [] # a list of tuples (unique_layer_index, nphi)
                occupancies_a_batch_edep = [] # a list of tuples (unique_layer_index, nphi, edep)
                # occupancies_a_batch_edep_per_cell = []
                batch_cell_fired_pos_neighbors = []
                batch_cell_fired_pos = []
                cell_to_mcID = [] #a list of tuples (unique_layer_index, nphi, mcID) for each cell fired for each batch
                cell_to_mcID_neighbors = [] #a list of tuples (unique_layer_index, nphi, mcID) for each cell fired for each batch
                all_mcID = [] #a list of mcID's which fired cells for each batch
                batch_pt = [] #will be a list of tuple of (radiusR, radiusPhi, pt)
                batch_pdg = [] #will be a list of tuple of (radiusR, radiusPhi, pdg)
                batch_low_pt = [] #will be a list of tuple of (radiusR, radiusPhi, pt)
                batch_high_pt = [] #will be a list of tuple of (radiusR, radiusPhi, pt)
                # numBatches = 0 #total batches should be numFiles * 10
                
            if typeFile == "combined": #want to reset every 1 combined event since for each file, 10 signal events, each with 20 bkg events respectively
                print(f"resetting occupancy for batch, new batch: {numBatches}")
                dict_cellID_nHits = {} #reset cells for every 20 bkg event
                occupancies_a_batch = []
                occupancies_a_batch_only_neighbor = [] # a list of tuples (unique_layer_index, nphi)
                occupancies_a_batch_edep = [] # a list of tuples (unique_layer_index, nphi, edep)
                # occupancies_a_batch_edep_per_cell = [] #a list of tuples (unique_layer_index, nphi, edep)
                occupancies_a_batch_isBkgOverlay = [] # a list of isBkgOverlay, should be same size as occupancies_a_batch
                # occupancies_a_batch_only_neighbor_isBkgOverlay = [] # a list of isBkgOverlay, should be same size as occupancies_a_batch_only_neighbor
                occupancies_a_batch_isSignalOverlay = [] # a list of isSignalOverlay, should be same size as occupancies_a_batch
                # occupancies_a_batch_only_neighbor_isSignalOverlay = [] # a list of isSignalOverlay, should be same size as occupancies_a_batch_only_neighbor
                batch_cell_fired_pos_neighbors = []
                batch_cell_fired_pos = []
                cell_to_mcID = [] #a list of tuples (unique_layer_index, nphi, mcID) for each cell fired for each batch
                cell_to_mcID_neighbors = [] #a list of tuples (unique_layer_index, nphi, mcID) for each cell fired for each batch
                all_mcID = [] #a list of mcID's which fired cells for each batch
                batch_pt = [] #will be a list of tuple of (radiusR, radiusPhi, pt)
                batch_pdg = [] #will be a list of tuple of (radiusR, radiusPhi, pdg)
                batch_low_pt = [] #will be a list of tuple of (radiusR, radiusPhi, pt)
                batch_high_pt = [] #will be a list of tuple of (radiusR, radiusPhi, pt)
        
            if typeFile == "combined":
                dc_hits = event.get("NewCDCHHits")
            else:
                dc_hits = event.get("DCHCollection")
            dict_particle_n_fired_cell = {}
            dict_particle_fired_cell_id = {}
            
            # print("starting hits")
            for num_hit, dc_hit in enumerate(dc_hits):
                # print(f"Running over hit: {num_hit}")
                mcParticleHit = dc_hit.getMCParticle()
                index_mc = mcParticleHit.getObjectID().index
                
                
                if index_mc not in all_mcID:
                    all_mcID.append(index_mc)
                
                if typeFile == "combined":
                    isBkgOverlay = dc_hit.isOverlay()
                    # print(isBkgOverlay)
                
                cellID = dc_hit.getCellID()
                superlayer = decoder.get(cellID, "superlayer")
                layer = decoder.get(cellID, "layer")
                nphi = decoder.get(cellID, "nphi")
                
                # define a unique layer index based on super layer and layer
                if layer >= n_layers_per_superlayer or superlayer >= n_superlayers:
                    print("Error: layer or super layer index out of range")
                    print(f"Layer: {layer} while max layer is {n_layers_per_superlayer - 1}. \
                        Superlayer: {superlayer} while max superlayer is {n_superlayers - 1}.")
                unique_layer_index = superlayer * n_layers_per_superlayer + layer #we will also use this as the radial index
                cellID_unique_identifier = "SL_" + str(superlayer)  + "_L_" + str(layer) + "_nphi_" + str(nphi) 
                
                # what is the occupancy?
                if not cellID_unique_identifier in dict_cellID_nHits.keys(): # the cell was not fired yet
                    occupancies_an_event.append(unique_layer_index)
                    occupancies_a_batch.append(unique_layer_index)
                    occupancies_a_batch_only_neighbor.append((unique_layer_index, nphi)) #we append for now then we can check after all hits, what neighbors are fired
                    occupancies_a_batch_edep.append((unique_layer_index, nphi, dc_hit.getEDep()))
                    # print(f"unique_layer_index: {unique_layer_index} and nphi: {nphi} where edep: {dc_hit.getEDep()}")
                    # occupancies_a_batch_edep_per_cell.append((unique_layer_index, nphi, dc_hit.getEDep()))
                    cellFiredPos.append((unique_layer_index, nphi))
                    cell_to_mcID.append((unique_layer_index, nphi, index_mc))
                    
                    
                    mcParticle = EventMCParticles[int(index_mc)]
                    pt = math.sqrt(mcParticle.getMomentum().x**2 + mcParticle.getMomentum().y**2)
                    batch_pt.append((radiusR, radiusPhi,  pt))
                    batch_pdg.append((radiusR, radiusPhi, mcParticle.getPDG()))
                    
                    # if pt < 0.01:
                    #     batch_low_pt.append((radiusR, radiusPhi, pt))
                    # else:
                    #     batch_high_pt.append((radiusR, radiusPhi, pt))
                    
                    
                    batch_cell_fired_pos.append((unique_layer_index, nphi))
                    dict_cellID_nHits[cellID_unique_identifier] = 1
                    if typeFile=="combined" and isBkgOverlay:
                        occupancies_a_batch_isBkgOverlay.append(unique_layer_index)
                    elif typeFile=="combined" and not isBkgOverlay:
                        occupancies_a_batch_isSignalOverlay.append(unique_layer_index)
                else: # the cell was already fired
                    dict_cellID_nHits[cellID_unique_identifier] += 1
                    cell_to_mcID.append((unique_layer_index, nphi, index_mc)) #still consider it
                    # print(f"unique_layer_index: {unique_layer_index} and nphi: {nphi}")
                    # find where in edep tuple the pos is of the previously fired cell
                    index = [i for i, tup in enumerate(occupancies_a_batch_edep) if tup[0] == unique_layer_index and tup[1] == nphi] 
                    #above can be otpimized if we use a dictionary but it goes fast as is
                    # print(i)
                    if len(index) == 0:
                        print("Error: unique_layer_index not found in occupancy_a_batch_edep")
                    else:
                        # print(f"occupancies_a_batch_edep[index[0]]: {occupancies_a_batch_edep[index[0]]}")
                        occupancies_a_batch_edep[index[0]] = ((unique_layer_index, nphi, occupancies_a_batch_edep[index[0]][2] + dc_hit.getEDep()))
                    
                # deal with the number of cell fired per particle
                if index_mc not in dict_particle_n_fired_cell.keys(): # the particle was not seen yet
                    dict_particle_n_fired_cell[index_mc] = 1
                    dict_particle_fired_cell_id[index_mc] = [cellID_unique_identifier]
                else: # the particle already fired cells
                    if not cellID_unique_identifier in dict_particle_fired_cell_id[index_mc]: # this cell was not yet fired by this particle
                        dict_particle_n_fired_cell[index_mc] += 1
                        dict_particle_fired_cell_id[index_mc].append(cellID_unique_identifier)
            ###end of hit loop
            
            ##signal files we want to reset every event
            if typeFile == "signal":
                print(f"setting occupancy for batch: {numBatches}")
                occupancies_per_batch_sum_batch[numBatches], \
                    occupancies_per_batch_sum_batch_only_neighbor[numBatches], \
                        occupancies_per_batch_sum_batch_energy_dep[numBatches], \
                            dic_occupancies_per_batch_sum_batch_energy_dep_per_cell, \
                                batch_cell_fired_pos_neighbors, cell_to_mcID_neighbors, \
                                    occupancies_per_batch_sum_batch_only_neighbors_only_edeps[numBatches], \
                                        edep_only_neighbors, edep_only_neighbors_only_edep, \
                                            cell_to_mcID_neighbors_edeps, \
                                                NoNeighborsRemoved, NeighborsRemained, \
                                                    EdepNeighborsRemained, NoEdepNeighborsRemoved = calcOcc(occupancies_a_batch, occupancies_a_batch_only_neighbor, 
                                                                                    occupancies_a_batch_edep, 
                                                                                    dic_occupancies_per_batch_sum_batch_energy_dep_per_cell, 
                                                                                    n_cell_per_layer, total_number_of_layers, 
                                                                                    radiusR, radiusPhi, atLeast, 
                                                                                    edepRange, edepAtLeast, max_n_cell_per_layer, cell_to_mcID,
                                                                                    NoNeighborsRemoved, NeighborsRemained, EdepNeighborsRemained, NoEdepNeighborsRemoved,
                                                                                    edepLoosen
                                                                                    )
                cellFiredPosNeighbors += batch_cell_fired_pos_neighbors
                cell_fired_pos_per_batch.append(batch_cell_fired_pos)
                cell_fired_pos_neighbors_per_batch.append(batch_cell_fired_pos_neighbors)
                energy_dep_per_cell_per_batch.append(occupancies_a_batch_edep) #a tuple of (unique_layer_index, nphi, edep)
                energy_dep_per_cell_per_batch_only_neighbors.append(edep_only_neighbors)
                energy_dep_per_cell_per_batch_only_neighbors_only_edeps.append(edep_only_neighbors_only_edep)
                all_mcID_per_batch.append(all_mcID)
                cell_to_mcID_per_batch.append(cell_to_mcID)
                cell_to_mcID_neighbors_per_batch.append(cell_to_mcID_neighbors)
                cell_to_mcID_neighbors_edeps_per_batch.append(cell_to_mcID_neighbors_edeps)
                neighborPt.append(batch_pt)
                neighborPDG.append(batch_pdg)
                # lowPt.append(batch_low_pt)
                # highPt.append(batch_high_pt)
                
                numBatches += 1
                
            if typeFile=="combined":
                print(f"setting occupancy for batch: {numBatches}")
                batch_occupancy_only_bkg = []
                batch_occupancy_only_signal = []
                for unique_layer_index in range(0, total_number_of_layers):
                    batch_occupancy_only_bkg.append(calculateOccupancy(occupancies_a_batch_isBkgOverlay, unique_layer_index, n_cell_per_layer))
                    batch_occupancy_only_signal.append(calculateOccupancy(occupancies_a_batch_isSignalOverlay, unique_layer_index, n_cell_per_layer))
                occupancies_per_batch_sum_batch_only_bkg[numBatches] = batch_occupancy_only_bkg
                occupancies_per_batch_sum_batch_only_signal[numBatches] = batch_occupancy_only_signal
                
                occupancies_per_batch_sum_batch[numBatches], \
                    occupancies_per_batch_sum_batch_only_neighbor[numBatches], \
                        occupancies_per_batch_sum_batch_energy_dep[numBatches], \
                            dic_occupancies_per_batch_sum_batch_energy_dep_per_cell, \
                                batch_cell_fired_pos_neighbors, cell_to_mcID_neighbors, \
                                    occupancies_per_batch_sum_batch_only_neighbors_only_edeps[numBatches], \
                                        edep_only_neighbors, edep_only_neighbors_only_edep, \
                                            cell_to_mcID_neighbors_edeps, \
                                                NoNeighborsRemoved, NeighborsRemained, \
                                                    EdepNeighborsRemained, NoEdepNeighborsRemoved = calcOcc(occupancies_a_batch, occupancies_a_batch_only_neighbor, 
                                                                                    occupancies_a_batch_edep, 
                                                                                    dic_occupancies_per_batch_sum_batch_energy_dep_per_cell, 
                                                                                    n_cell_per_layer, total_number_of_layers, 
                                                                                    radiusR, radiusPhi, atLeast, 
                                                                                    edepRange, edepAtLeast, max_n_cell_per_layer, cell_to_mcID,
                                                                                    NoNeighborsRemoved, NeighborsRemained, EdepNeighborsRemained, NoEdepNeighborsRemoved,
                                                                                    edepLoosen
                                                                                    )
                cellFiredPosNeighbors += batch_cell_fired_pos_neighbors
                cell_fired_pos_per_batch.append(batch_cell_fired_pos)
                cell_fired_pos_neighbors_per_batch.append(batch_cell_fired_pos_neighbors)
                energy_dep_per_cell_per_batch.append(occupancies_a_batch_edep) #a tuple of (unique_layer_index, nphi, edep)
                energy_dep_per_cell_per_batch_only_neighbors.append(edep_only_neighbors)
                energy_dep_per_cell_per_batch_only_neighbors_only_edeps.append(edep_only_neighbors_only_edep)
                all_mcID_per_batch.append(all_mcID)
                cell_to_mcID_per_batch.append(cell_to_mcID)
                cell_to_mcID_neighbors_per_batch.append(cell_to_mcID_neighbors)
                cell_to_mcID_neighbors_edeps_per_batch.append(cell_to_mcID_neighbors_edeps)
                neighborPt.append(batch_pt)
                neighborPDG.append(batch_pdg)
                # lowPt.append(batch_low_pt)
                # highPt.append(batch_high_pt)
                
                numBatches += 1
        ###end of event loop
            
            
        #the next one resets the 20 file batch
        if (i + 1) % batches == 0 and typeFile == "bkg":
            print(f"setting occupancy for batch: {numBatches}")
            occupancies_per_batch_sum_batch[numBatches], \
                occupancies_per_batch_sum_batch_only_neighbor[numBatches], \
                    occupancies_per_batch_sum_batch_energy_dep[numBatches], \
                        dic_occupancies_per_batch_sum_batch_energy_dep_per_cell, \
                            batch_cell_fired_pos_neighbors, cell_to_mcID_neighbors, \
                                occupancies_per_batch_sum_batch_only_neighbors_only_edeps[numBatches], \
                                    edep_only_neighbors, edep_only_neighbors_only_edep, \
                                        cell_to_mcID_neighbors_edeps, \
                                            NoNeighborsRemoved, NeighborsRemained, \
                                                EdepNeighborsRemained, NoEdepNeighborsRemoved = calcOcc(occupancies_a_batch, occupancies_a_batch_only_neighbor, 
                                                                                    occupancies_a_batch_edep, 
                                                                                    dic_occupancies_per_batch_sum_batch_energy_dep_per_cell, 
                                                                                    n_cell_per_layer, total_number_of_layers, 
                                                                                    radiusR, radiusPhi, atLeast, 
                                                                                    edepRange, edepAtLeast, max_n_cell_per_layer, cell_to_mcID,
                                                                                    NoNeighborsRemoved, NeighborsRemained, EdepNeighborsRemained, NoEdepNeighborsRemoved,
                                                                                    edepLoosen
                                                                                    )
            cellFiredPosNeighbors += batch_cell_fired_pos_neighbors
            cell_fired_pos_per_batch.append(batch_cell_fired_pos)
            cell_fired_pos_neighbors_per_batch.append(batch_cell_fired_pos_neighbors)
            energy_dep_per_cell_per_batch.append(occupancies_a_batch_edep) #a tuple of (unique_layer_index, nphi, edep)
            energy_dep_per_cell_per_batch_only_neighbors.append(edep_only_neighbors)
            energy_dep_per_cell_per_batch_only_neighbors_only_edeps.append(edep_only_neighbors_only_edep)
            all_mcID_per_batch.append(all_mcID)
            cell_to_mcID_per_batch.append(cell_to_mcID)
            cell_to_mcID_neighbors_per_batch.append(cell_to_mcID_neighbors)
            cell_to_mcID_neighbors_edeps_per_batch.append(cell_to_mcID_neighbors_edeps)
            neighborPt.append(batch_pt)
            neighborPDG.append(batch_pdg)
            # lowPt.append(batch_low_pt)
            # highPt.append(batch_high_pt)
            
            numBatches += 1
            
        
        
        
        for particleKey in dict_particle_n_fired_cell.keys():
            list_n_cells_fired_mc.append(dict_particle_n_fired_cell[particleKey])
                    
                    
        # percentage_of_fired_cells.append(100 * len(dict_cellID_nHits.keys())/float(total_number_of_cells)  )
    ###end of file loop

    print("end of file loop")


    print("updating dictionary")
    #at this point we have filled the occupancy so there are 500 rows and 112 columns

    dic["occupancy_per_batch_sum_batches"] = np.mean(occupancies_per_batch_sum_batch, axis=0)
    dic["occupancy_per_batch_sum_batches_error"] = np.std(occupancies_per_batch_sum_batch, axis=0)
    dic["occupancy_per_batch_sum_batches_non_meaned"] = occupancies_per_batch_sum_batch

    dic["occupancy_per_batch_sum_batches_only_neighbor"] = np.mean(occupancies_per_batch_sum_batch_only_neighbor, axis=0)
    dic["occupancy_per_batch_sum_batches_only_neighbor_error"] = np.std(occupancies_per_batch_sum_batch_only_neighbor, axis=0)
    # print(f"no neighbors removed: {NoNeighborsRemoved}")
    # print(f"remained neighbors: {NeighborsRemained}")
    dic["no_neighbors_removed"] = NoNeighborsRemoved
    dic["neighbors_remained"] = NeighborsRemained
    
    print(f"no edep neighbors removed: {NoEdepNeighborsRemoved}")
    dic["no_edep_neighbors_removed"] = NoEdepNeighborsRemoved
    dic["edep_neighbors_remained"] = EdepNeighborsRemained
    
    dic["occupancy_per_batch_sum_batches_only_neighbor_only_edep"] = np.mean(occupancies_per_batch_sum_batch_only_neighbors_only_edeps, axis=0)
    dic["occupancy_per_batch_sum_batches_only_neighbor_only_edep_error"] = np.std(occupancies_per_batch_sum_batch_only_neighbors_only_edeps, axis=0)

    #given the dictionary of key to list of edep, we will now mean the list of edep
    dic["dic_occupancy_per_batch_sum_batches_energy_dep"] = {}
    for key in dic_occupancies_per_batch_sum_batch_energy_dep_per_cell.keys():
        dic["dic_occupancy_per_batch_sum_batches_energy_dep"][key] = np.mean(dic_occupancies_per_batch_sum_batch_energy_dep_per_cell[key])

    dic["energy_dep_per_cell_per_batch"] = energy_dep_per_cell_per_batch
    dic["energy_dep_per_cell_per_batch_only_neighbors"] = energy_dep_per_cell_per_batch_only_neighbors
    dic["energy_dep_per_cell_per_batch_only_neighbors_only_edeps"] = energy_dep_per_cell_per_batch_only_neighbors_only_edeps
        
    dic["occupancy_per_batch_sum_batches_only_bkg"] = np.mean(occupancies_per_batch_sum_batch_only_bkg, axis=0)
    dic["occupancy_per_batch_sum_batches_only_bkg_error"] = np.std(occupancies_per_batch_sum_batch_only_bkg, axis=0)
    dic["occupancy_per_batch_sum_batches_only_signal"] = np.mean(occupancies_per_batch_sum_batch_only_signal, axis=0)
    dic["occupancy_per_batch_sum_batches_only_signal_error"] = np.std(occupancies_per_batch_sum_batch_only_signal, axis=0)

    dic["occupancy_per_batch_sum_batch_avg_energy_dep"] = np.mean(occupancies_per_batch_sum_batch_energy_dep, axis=0)
    dic["occupancy_per_batch_sum_batch_avg_energy_dep_error"] = np.std(occupancies_per_batch_sum_batch_energy_dep, axis=0)
    dic["cell_fired_pos"] = cellFiredPos
    dic["cell_fired_pos_neighbors"] = cellFiredPosNeighbors
    dic["cell_fired_pos_by_batch"] = cell_fired_pos_per_batch
    
    dic["onlyNeighborMCID_per_batch"] = cell_to_mcID_neighbors_per_batch
    dic["onlyNeighborOnlyEdepMCID_per_batch"] = cell_to_mcID_neighbors_edeps_per_batch
    dic["cellFiredMCID_per_batch"] = cell_to_mcID_per_batch
    
    dic["neighborPt_by_batch"] = neighborPt
    dic["neighborPDG_by_batch"] = neighborPDG

    # print(f"shape of occupancy_per_batch_sum_batches: {dic['occupancy_per_batch_sum_batches'].shape}")

    # dic["percentage_of_fired_cells"] += percentage_of_fired_cells
    dic["list_n_cells_fired_mc"] += list_n_cells_fired_mc
    dic["n_cell_per_layer"] = n_cell_per_layer
    dic["total_number_of_cells"] = total_number_of_cells
    dic["total_number_of_layers"] = total_number_of_layers
    dic["max_n_cell_per_layer"] = max_n_cell_per_layer #this is max phi index!!!


    print(f"No neighbors removed: {NoNeighborsRemoved}")
    print(f"Neighbors remained: {NeighborsRemained}")
    print(f"No edep neighbors removed: {NoEdepNeighborsRemoved}")
    print(f"Edep neighbors remained: {EdepNeighborsRemained}")

    print(f"Saving dictionary to {output_dic_file_path}")
    np.save(output_dic_file_path, dic)


if __name__ == "__main__":
    #create argument parser so someone can create start dat without hard coding
    parser = argparse.ArgumentParser()
    typeFile = ["bkg", "signal", "combined"]
    parser.add_argument('--calc', help="Inputs... " +
                        "\n-- fileType(str): [bkg], [signal], [combined] Default(bkg)" +
                        "\n-- numfiles(int): Default(500)" +
                        "\n-- radiusR(int): Default(1)" +
                        "\n-- radiusPhi(int): Default(-1)" +
                        "\n-- atLeast(int): Default(1)" +
                        "\n-- edepRange(float): Default(0.05)" + 
                        "\n-- edepAtLeast(int): Default(1)" +
                        "\n-- edepLoosen(bool): Default(False)",
                        type=str, default="", nargs='+')
    args = parser.parse_args()
    
    starttime = time.time()

    if args.calc and args.calc != "":
        try:
            print(f"Parsed --calc arguments: {args.calc}")
            if args.calc[0] in typeFile and len(args.calc) == 1:
                updateOcc(args.calc[0])
            elif args.calc[0] in typeFile and len(args.calc) == 2:
                updateOcc(args.calc[0], int(args.calc[1]))
            elif args.calc[0] in typeFile and len(args.calc) == 3:
                updateOcc(args.calc[0], int(args.calc[1]), int(args.calc[2]))
            elif args.calc[0] in typeFile and len(args.calc) == 4:
                updateOcc(args.calc[0], int(args.calc[1]), int(args.calc[2]), int(args.calc[3]))
            elif args.calc[0] in typeFile and len(args.calc) == 5:
                updateOcc(args.calc[0], int(args.calc[1]), int(args.calc[2]), int(args.calc[3]), int(args.calc[4]))
            elif args.calc[0] in typeFile and len(args.calc) == 6:
                updateOcc(args.calc[0], int(args.calc[1]), int(args.calc[2]), int(args.calc[3]), int(args.calc[4]), float(args.calc[5]))
            elif args.calc[0] in typeFile and len(args.calc) == 7:
                updateOcc(args.calc[0], int(args.calc[1]), int(args.calc[2]), int(args.calc[3]), int(args.calc[4]), float(args.calc[5]), int(args.calc[6]))
            elif args.calc[0] in typeFile and len(args.calc) == 8:
                updateOcc(args.calc[0], int(args.calc[1]), int(args.calc[2]), int(args.calc[3]), int(args.calc[4]), float(args.calc[5]), int(args.calc[6]), bool(args.calc[7]))
            else:
                parser.error("Invalid fileType")
        except ValueError as e:
            parser.error(str(e))
            
    endtime = time.time()
    print("Time taken: ", endtime - starttime)
    #'''