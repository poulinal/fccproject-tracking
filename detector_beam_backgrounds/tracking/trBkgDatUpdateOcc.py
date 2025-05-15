#Alexander Poulin Jan 2025
from podio import root_io
import numpy as np 
import math
import dd4hep as dd4hepModule
from ROOT import dd4hep
import sys
from trBkgDat import configure_paths, setUpFiles
import argparse
import time
import os
from pyDCH_info import DCH_info
from utilities.utils import check_odd_fractions, globalPhiIndex, find_closest_indices, kappa

"""
This script is used to update the occupancy of the background particles in the dictionary.
This can be run standalone just make sure to update filePaths accordingly.
This is meant to be ran after trBkgDat.py (which save a .npy)
but doesnt have to, it will just create a new .npy.
"""


def setupBatchVars(typeFile, numBatches, particleType):
    print(f"resetting occupancy for batch, new batch: {numBatches}") 
    batchVars = {}
    batchVarsKeys = ["dict_cellID_nHits",
                     "dict_cellID_nHits_full", 
                       "pos_only_neighbors",
                       "pos_only_neighbors_only_edeps",
                       "pos" ,
                       #"occupancies_a_batch_only_neighbor", 
                       #"batch_edep",
                       #"occupancies_a_batch_edep_per_cell", 
                       #"batch_cell_fired_pos_neighbors", 
                       #"batch_pos",
                       #"cell_to_mcID", #"cell_to_mcID_neighbors", 
                       #"batch_pt", "batch_pdg",
                       #"batch_prod_sec", "batch_photon_par", "batch_cart_pos_status"]
                        ]
    # for key in batchedCellFiredVarsKeys:
    #     batchedCellFiredVars[key] = {}
    batchVars["dict_cellID_nHits"] = {} #dict unique identifier to num hits
    batchVars["dict_cellID_nHits_full"] = {} #dict unique identifier to num hits for phi, r, z
    batchVars["pos"] = [] #a list of tuples (unique_layer_index, nphi)
    batchVars["pos_only_neighbors"] = [] #a list of tuples (unique_layer_index, nphi)
    batchVars["pos_only_neighbors_only_edeps"] = [] #a list of tuples (unique_layer_index, nphi)
    batchVars["pos_bkg_overlay"] = [] #a list of tuples (unique_layer_index, nphi, isBkgOverlay)
    batchVars["pos_signal_overlay"] = [] #a list of tuples (unique_layer_index, nphi, isBkgOverlay)
    batchVars['particleType'] = particleType
    # dict_cellID_nHits = {} #reset cells for every 20 bkg event
    # occupancies_a_batch = [] # a list of tuples (unique_layer_index, nphi)
    # occupancies_a_batch_edep = [] # a list of tuples (unique_layer_index, edep)
    # # occupancies_a_batch_edep_per_cell = [] #a list of tuples (unique_layer_index, nphi, edep)
    # batch_cell_fired_pos_neighbors = []
    # batch_cell_fired_pos = []
    
    # cell_to_mcID = {} #a dic key is layer,phi value is a list of tuples (unique_layer_index, nphi, mcID) for each cell fired for each batch
    # cell_to_mcID_neighbors = {} #a dic key is layer,phi value is a list of tuples (unique_layer_index, nphi, mcID) for each cell fired for each batch
    # batch_pt = [] #will be a list of tuple of (radiusR, radiusPhi, pt)
    # batch_pdg = [] #will be a list of tuple of (radiusR, radiusPhi, pdg)
    # batch_prod_sec = {} #will be a dic of tuple of pos to (radiusR, radiusPhi, prod_sec, MCID)
    # batch_photon_par = {} #will be a dic of tuple of pos to (radiusR, radiusPhi, has_parent_photon, MCID)
    # batch_cart_pos_status = {} #will be a dic of tuple of pos to (radiusR, radiusPhi, x, y, z, genstatus, MCID)
    # particleType = 1 #1 for bkg
    return batchVars

def driftChamberProperties():
    # Drift chamber geometry parameters
    n_layers_per_superlayer = 8
    n_superlayers = 14
    total_number_of_layers = 0
    n_cell_superlayer0 = 192
    n_cell_increment = 48
    n_cell_per_layer = {}
    n_cell_per_superlayer = {}
    total_number_of_cells = 0
    for sl in range(0, n_superlayers):
        for l in range(0, n_layers_per_superlayer):
            total_number_of_layers += 1
            total_number_of_cells += n_cell_superlayer0 + sl * n_cell_increment
            n_cell_per_layer[str(n_layers_per_superlayer * sl + l)] = n_cell_superlayer0 + sl * n_cell_increment
        n_cell_per_superlayer[str(sl)] = n_cell_superlayer0 + sl * n_cell_increment
    print("total_number_of_cells: ", total_number_of_cells)
    print("total_number_of_layers: ", total_number_of_layers)
    print("n_cell_per_layer: ", n_cell_per_layer)
    max_n_cell_per_layer = n_cell_per_layer[str(total_number_of_layers - 1)]
    
    list_max_n_cell_per_layer = []
    for key in n_cell_per_layer:
        list_max_n_cell_per_layer.append(n_cell_per_layer[key])
    
    return total_number_of_layers, n_cell_per_layer, n_cell_per_superlayer, max_n_cell_per_layer, total_number_of_cells, n_layers_per_superlayer, n_superlayers, list_max_n_cell_per_layer
    
def stereoWire(p_c, z, phi, k, phi_d, L):
    """Given the radius of the wire, z position of the cell, angle of the wire about the axis, and kappa of the wire, this function returns the new coordinates of the wire and the angle theta based on a twisted surface.

    Args:
        p_c (int): radius of the wire (z=0) (currently just index of layer... shouldnt have an effect)
        z (float): z position of the cell
        phi (float): angle of the wire about the axis (z=0)
        k (float): kappa of the wire (constant based on length and twist angle)
        phi_d (float): twist angle of the wire (in radians)

    Returns:
        _type_: _description_
    """
    
    x_new = p_c * np.cos(phi) - p_c*k*z*np.sin(phi)
    y_new = p_c * np.sin(phi) + p_c*k*z*np.cos(phi)
    t_r = (x_new, y_new, z)
    
    arc_length_shift = 2 * p_c * z / L * np.tan(phi_d / 2) #arc length along same p_c compared to z=0
    
    theta = np.arctan2(y_new, x_new)
    return t_r, theta, arc_length_shift

def zsteps(DCHi):
    """Generates the z steps for the drift chamber. This is done by looping over the z positions and calculating the t_r and theta for each layer. Currently set to 100 steps length.

    Args:
        None

    Returns:
        dict: dictionary which maps z position to a list of tuples (layer, t_r, theta) for each layer
    """
    
    stop = int(DCHi.lhalf)
    start = -int(DCHi.lhalf)
    step = 100
    zpos = np.arange(start, stop + step, step)  # Ensure stop is inclusive

    return zpos

def layerDrift(DCHi, n_cell_per_layer):
    """Generates the phi drift for a given layer and z position. This is done by looping over the z positions and calculating the t_r and theta for each layer.

    Args:
        None

    Returns:
        dict: dictionary which maps z position to a list of tuples (layer, t_r, theta) for each layer
    """
    
    phi_d = DCHi.twist_angle * np.pi / 180 #twist angle
    L = DCHi.lhalf * 2 #half length * 2
    
    # total_layers = int(DCHi.nlayersPerSuperlayer * DCHi.nsuperlayers)
    k = kappa(L, phi_d)
    
    zpos = zsteps(DCHi)
    # print(f"n_cell_per_layer: {n_cell_per_layer}")
    # print(f"zpos: {zpos}")
    # input("Press Enter to continue... zpos")

    z_layer_to_shift = {}
    for i, z in enumerate(zpos):
        layer_to_shift = []
        
        for j in range(0, len(DCHi._database)):
            p_c = DCHi._database[j]['radius_sw_z0'] #radius of the wire at z=0
            stereoSign = DCHi._database[j]['stereo_sign']
            t_r, theta, arc_length_shift = stereoWire(p_c, z, 0, k*stereoSign, phi_d, L) #same for all layers
            
            arc_length_step = 2 * np.pi * z / n_cell_per_layer[str(j)] #arc length step for the layer
            layer_to_shift.append((j, p_c, t_r, arc_length_shift, arc_length_step))
            ### (layer, radius, (x_new, y_new, z), arc_length_shift, arc_length_step) ###
            # print(f"layer: {j}, t_r: {t_r}, theta: {theta}, stereoSign: {stereoSign}, arc_length_shift: {arc_length_shift * stereoSign}")
            #get circumference of p_c
        #     print(f"circumference: {2 * np.pi * p_c}")
        # input("Press Enter to continue... DCHi layers")
        z_layer_to_shift[z] = layer_to_shift
        # t_r, theta = stereoWire(350, z, 0, k)
    # all_arc_length_shifts = [layer[3] for z in z_layer_to_shift.values() for layer in z]
    # print(f"all arc length shifts: {all_arc_length_shifts}")
    # all_arc_length_steps = [layer[4] for z in z_layer_to_shift.values() for layer in z]
    # print(f"all arc length steps: {all_arc_length_steps}")
    # input("Press Enter to continue... z_layer_to_shift")
    
        
    return z_layer_to_shift

def calculateOccupancy(occupancy :list[tuple], unique_layer_index, n_cell_per_layer):
    #basicaly, we are calculating the occupancy of each layer
    #so for each layer, we get the number of cells that were fired and divide by the total number of cells in that layer
    #occupancy is a list of tuples (unique_layer_index, nphi)
    onlyLayers = [x[0] for x in occupancy]
    # print(f"onlyLayers: {onlyLayers}")
    filtered_occupancies = [x for x in onlyLayers if x == unique_layer_index]
    # print(f"filtered_occupancies: {filtered_occupancies}")
    layer_count = len(filtered_occupancies)
    total_cells_in_layer = float(n_cell_per_layer[str(unique_layer_index)])
    percentage_occupancy = 100 * layer_count / total_cells_in_layer
    return percentage_occupancy

def calculateOccupancyRawCount(occupancy, unique_layer_index, n_cell_per_layer):
    #calculate occupancy but just number of fired cells per layer pretty much
    filtered_occupancies = [x for x in occupancy if x == unique_layer_index]
    layer_count = len(filtered_occupancies)
    return layer_count

def calculateOnlyNeighbors(dic_int_vars, dic_posToKey_by_batch, dic_RPhiKey_by_batch, dic_list_mcIDs_one_batch, dic_all_occupancies, batchVars, maxLayer=112):
    #calculate the occupancy of non-neighbor cells
    #we will loop over all the cells and check if they have neighbors
    #a neightbor will be defined if there exists an occupancy index where (unique_layer_index +-0 or 1, nphi +- 0 or 1) exists
    #if they do, we will remove them from the list
    #occupancy is a list of tuples (unique_layer_index, nphi)
    #we will return a list of unique_layer_index
    
    radiusR = dic_int_vars['radiusR']
    radiusPhi = dic_int_vars['radiusPhi']
    atLeast = dic_int_vars['atLeast']
    edepRange = dic_int_vars['edepRange']
    edepAtLeast = dic_int_vars['edepAtLeast']
    edepLoosen = dic_int_vars['edepLoosen']
    zrange = dic_int_vars['zrange']
    
    NoNeighborsRemoved = dic_int_vars['NoNeighborsRemoved']
    NeighborsRemained = dic_int_vars['NeighborsRemained']
    EdepNeighborsRemained = dic_int_vars['EdepNeighborsRemained']
    NoEdepNeighborsRemoved = dic_int_vars['NoEdepNeighborsRemoved']
    
    maxnphiPerSuperLayer = dic_int_vars['list_max_n_cell_per_superlayer']
    maxnphiPerLayer = dic_int_vars['list_max_n_cell_per_layer']
    
    z_layer_to_shift = dic_int_vars['z_layer_to_shift']
    # print(f"z_layer_to_shift: {z_layer_to_shift}")
    
    dicNeighbors = {} #will be a dictionary where key is pos of some cell fired, the value will be a list of neighbor pos
    dicEdepNeighbors = {} #setup dictionary for current cell's neighbors edep
    
    if radiusPhi == -1:
        radiusPhi = radiusR
    # print(f"calculateNNOcc: {np.array(occupancy)}")
    # print(f"rangeR: {rangeR}, rangePhi: {rangePhi}")
    for i, key in enumerate(list(dic_posToKey_by_batch.keys())): #where i is the layer number
        unique_layer_index = key[0]
        superLayerIndex = unique_layer_index // 8 #indexes at 0
        nphi = key[1]
        hit_z = key[2] #already defined in closest z to zpos
        
        n_cells_in_layer = maxnphiPerSuperLayer[superLayerIndex] #number of cells in the layer
 
        parent_shift_info = z_layer_to_shift[hit_z][unique_layer_index] #get the shift info for the layer
        parent_stereo_sign = 1 if parent_shift_info[3] >= 0 else 0
        
        # current_pos = (unique_layer_index, nphi)
        current_pos_full = (unique_layer_index, nphi, hit_z)
        currentEdep = dic_posToKey_by_batch[current_pos_full]['energy_dep_per_cell_RPhiZ'][0] if len(dic_posToKey_by_batch[current_pos_full]['energy_dep_per_cell_RPhiZ']) == 1 else RuntimeError("check since more than one edep") #energy deposition in the cell if we take edep for entire wire
        # currentEdep = dic_posToKey_by_batch[current_pos_full]['energy_dep_per_cell'][0] if len(dic_posToKey_by_batch[current_pos_full]['energy_dep_per_cell']) == 1 else RuntimeError("check since more than one edep") #energy deposition in the cell if we take edep for some specific z
        
        
        superLayerIndex = (unique_layer_index) // 8 #indexes at 0
        maxnphi = maxnphiPerLayer[unique_layer_index] #number of cells in the layer
        
        neighborAtLeast = atLeast
        edepNeighborAtLeast = edepAtLeast
        rangeR = radiusR
        rangePhi = radiusPhi
        rangeZ = zrange
        if edepLoosen and superLayerIndex > 11: # maxLayer / 8 / 8: #since 8 layers per superlayer; start loosening after second to last superlayer
            layerSinceHalf = max(0, (unique_layer_index - (11 * 8)))
            #superLayerSinceHalf = superLayerIndex - maxLayer / 8 / 8
            # neighborAtLeast = int(atLeast * pow(0.99, layerSinceHalf))
            # edepNeighborAtLeast = int(edepAtLeast * pow(1.05, layerSinceHalf))
            # edepNeighborAtLeast = edepAtLeast
            rangeR = int(radiusR * pow(1.025, layerSinceHalf))
            rangePhi = int(radiusPhi * pow(1.025, layerSinceHalf)) #take into account the superlayer into the adjustment
            
            
        parent_layer_arc_length_step = parent_shift_info[4] #arc length step for the layer
        
        #need to check if each neighbor_zlayer shift and parent_zlayer shift are >= 1/4 layer phi step respectively; since radially swepted, both will cross their respective 1/4 phi step at the same time despite the different z positions and radial out
        #this assumption should hold for the same superlayer
        check_parent_shift_indicie = check_odd_fractions(parent_shift_info[3], parent_layer_arc_length_step) #check if the shift is greater than 1/4 layer phi step
        
        parent_phi_indicie_shift = check_parent_shift_indicie[1] if check_parent_shift_indicie[0] else -1 #this is the indicie of the shift
        
        
        neighbors = False
        neighborsEdep = False
        numNeighbors = 0
        numEdepNeighbors = 0
        if current_pos_full not in dicNeighbors: #setup current cell if not seen before
            dicNeighbors[current_pos_full] = [] #setup dictionary for current cell's neighbors
        if current_pos_full not in dicEdepNeighbors:
            dicEdepNeighbors[current_pos_full] = [] #setup dictionary for current cell's neighbors edep
            
            
        #based on the z and r, we can determine what phi indicies to look at
        #for now we will look at global ranges, but we can also look at relative ranges/indicies later
        for dradius in range(-rangeR, rangeR + 1):
            if len(dicNeighbors[current_pos_full]) >= neighborAtLeast and len(dicEdepNeighbors[current_pos_full]) >= edepNeighborAtLeast: #have we already seen enough
                neighbors = True
                neighborsEdep = True
                break #skip if we have already seen enough neighbors and break out of dx loop
            
            if dradius == 0:
                continue
            for dz in range(-rangeZ, rangeZ + 1, dic_int_vars['zStep']):
                neighbor_zlayer_candidate_info = z_layer_to_shift[hit_z + dz][unique_layer_index] #get the shift info for the layer
                
                neighbor_zlayer_candidate_stereo_sign = 1 if neighbor_zlayer_candidate_info[3] >= 0 else 0
                
                if neighbor_zlayer_candidate_stereo_sign != parent_stereo_sign: #check if the stereo sign is the same
                    neighbor_zlayer_shift = neighbor_zlayer_candidate_info[3]#arc length shift for the layer
                    check_neighbor_zlayer_shift_indicie = check_odd_fractions(neighbor_zlayer_shift, parent_layer_arc_length_step) #check if the shift is greater than 1/4 layer phi step
                    
                    neighbor_zlayer_shift_indicie = check_neighbor_zlayer_shift_indicie[1] if check_neighbor_zlayer_shift_indicie[0] else -1 #this is the indicie of the shift
                else:
                    neighbor_zlayer_shift, parent_phi_indicie_shift = 0, 0 #i.e. if the stereo sign is the same, we can assume the shift is 0 since they are the same layer and will be shifting in parallel
                    
                if neighbor_zlayer_shift_indicie != parent_phi_indicie_shift:
                    print(f"neighbor_zlayer_shift_indicie: {neighbor_zlayer_shift_indicie}, parent_phi_indicie_shift: {parent_phi_indicie_shift[1]}")
                    input("Press Enter to continue... neighbor_zlayer_shift_indicie")
                else: #should be able to use the shift indicies
                    for dphi in range(-rangePhi, rangePhi + 1):
                        # print(f"dx: {dx}, dy: {dy}")
                        if dphi == 0 and dradius == 0 and dz == 0: # Skip the center point
                            continue
                        if dradius == 0:  #skip same layer
                            continue
                        # print(f"phi: {phi}, radius: {radius}, z: {z}")
                        # print(f"neighbor_zlayer_shift_indicie: {neighbor_zlayer_shift_indicie}, parent_phi_indicie_shift: {parent_phi_indicie_shift[1]}")
                        # input("Press Enter to continue... neighbor_zlayer_shift_indicie")
                        #check if the indicies are the same
                        if neighbor_zlayer_shift_indicie != parent_phi_indicie_shift:
                            print(f"Error: neighbor_zlayer_shift_indicie: {neighbor_zlayer_shift_indicie}, parent_phi_indicie_shift: {parent_phi_indicie_shift[1]}")
                            input("Press Enter to continue... neighbor_zlayer_shift_indicie")
                            
                        cyclic_nphi = ((nphi + dphi) + neighbor_zlayer_shift_indicie) % maxnphi  # Wrap around for cyclic nphi #we will assume 180 for now
                        cyclic_unique_layer_index = unique_layer_index + dradius
                        noncyclic_z = hit_z + dz
                        
                        if cyclic_unique_layer_index < 0 or cyclic_unique_layer_index >= maxLayer: #check boundaries ###fixx
                            continue
                        
                        # neighbor_pos = (cyclic_unique_layer_index, cyclic_nphi)
                        neighbor_pos_full = (cyclic_unique_layer_index, cyclic_nphi, noncyclic_z)
                        
                        neighbor_shift_info = z_layer_to_shift[noncyclic_z][cyclic_unique_layer_index] #get the shift info for the layer
                        
                        ### if neighbor_pos in dic_RPhiKey_by_batch[pos_full]:
                        
                        if len(dicNeighbors[current_pos_full]) < neighborAtLeast and neighbor_pos_full in dic_posToKey_by_batch and neighbor_pos_full not in dicNeighbors[current_pos_full]: 
                            #not already over nieghbor atleast
                            #nieghbor exists (i.e. has been fired) (then assume also exists in edep)
                            #and it hasnt already been counted in dicNeighbors
                            numNeighbors += 1
                            
                            dicNeighbors[current_pos_full].append(neighbor_pos_full) #add neighbor to cell
                            if neighbor_pos_full not in dicNeighbors:
                                dicNeighbors[neighbor_pos_full] = []
                            dicNeighbors[neighbor_pos_full].append(current_pos_full) #add cell to neighbor (reduce double counting)
                            
                        if len(dicEdepNeighbors[current_pos_full]) < edepNeighborAtLeast and neighbor_pos_full in dic_posToKey_by_batch and neighbor_pos_full not in dicEdepNeighbors[current_pos_full]:
                            neighborEdep = dic_posToKey_by_batch[neighbor_pos_full]['energy_dep_per_cell_RPhiZ'][0] if len(dic_posToKey_by_batch[neighbor_pos_full]['energy_dep_per_cell_RPhiZ']) == 1 else RuntimeError("check since more than one edep") #energy deposition in the cell
                            # print(f"edeps: {currentEdep}, {neighborEdep}")
                            if abs(currentEdep - neighborEdep) <= edepRange: #if neighbor within range of edep
                                numEdepNeighbors += 1
                                dicEdepNeighbors[current_pos_full].append(neighbor_pos_full) #add neighbor to cell
                                if neighbor_pos_full not in dicEdepNeighbors:
                                    dicEdepNeighbors[neighbor_pos_full] = []
                                dicEdepNeighbors[neighbor_pos_full].append(current_pos_full) #add cell to neighbor (reduce double counting)
                                
                        if numNeighbors >= neighborAtLeast:
                            neighbors = True
                        if numEdepNeighbors >= edepNeighborAtLeast:
                            neighborsEdep = True
                        if neighborsEdep and neighbors: #this may be redundant due to first check
                            # print("break early")
                            break
                

        #determine outcome of cell:
        if neighbors: #if neighbors, add to only_neighbors
            batchVars['pos_only_neighbors'].append((unique_layer_index, nphi, hit_z)) #also should do edep for only neighbors only edep
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
            batchVars['pos_only_neighbors_only_edeps'].append((unique_layer_index, nphi, hit_z))
            if type(EdepNeighborsRemained) == int:
                EdepNeighborsRemained += 1
            else:
                EdepNeighborsRemained.value += 1
        else:
            if type(NoEdepNeighborsRemoved) == int:
                NoEdepNeighborsRemoved += 1
            else:
                NoEdepNeighborsRemoved.value += 1    
    print(f"no neighbors removed: {NoNeighborsRemoved}")
    print(f"no edep neighbors removed: {NoEdepNeighborsRemoved}")
    
    dic_int_vars['NoNeighborsRemoved'] = NoNeighborsRemoved
    dic_int_vars['NeighborsRemained'] = NeighborsRemained
    dic_int_vars['EdepNeighborsRemained'] = EdepNeighborsRemained
    dic_int_vars['NoEdepNeighborsRemoved'] = NoEdepNeighborsRemoved
    
    #currently not returning NoNeighborsRemoved so its just a copy right now, the final value will not be correct
    return dic_int_vars, dic_posToKey_by_batch, dic_RPhiKey_by_batch, dic_list_mcIDs_one_batch, dic_all_occupancies, batchVars 
        
def calcOcc(dic_int_vars, dic_posToKey_by_batch, dic_RPhiKey_by_batch, dic_list_mcIDs_one_batch, dic_all_occupancies, batchVars):
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
    for unique_layer_index in range(0, dic_int_vars['total_number_of_layers']):
        batch_occupancy.append(calculateOccupancy(batchVars['pos'], unique_layer_index, dic_int_vars['n_cell_per_layer']))
    # occupancies_per_batch_sum_batch[numBatches] = batch_occupancy #note index should be fileNum + eventNum / batches(1)
    
    #now determine non-neighbor occupancy
    batch_occupancy_only_neighbor = []
    batch_occupancy_only_neighbor_only_edep = []
    dic_int_vars, dic_posToKey_by_batch, dic_RPhiKey_by_batch, dic_list_mcIDs_one_batch, dic_all_occupancies, batchVars = calculateOnlyNeighbors(dic_int_vars, dic_posToKey_by_batch, dic_RPhiKey_by_batch, dic_list_mcIDs_one_batch, dic_all_occupancies, batchVars) 
    #for each batch, occupancies_a_batch_only_neighbor is a list of tuples but we will return a list of unique_layer_index
    for unique_layer_index in range(0, dic_int_vars['total_number_of_layers']):
        batch_occupancy_only_neighbor.append(calculateOccupancy(batchVars['pos_only_neighbors'], unique_layer_index, dic_int_vars['n_cell_per_layer']))
        batch_occupancy_only_neighbor_only_edep.append(calculateOccupancy(batchVars['pos_only_neighbors_only_edeps'], unique_layer_index, dic_int_vars['n_cell_per_layer']))
    dic_all_occupancies['occupancies_per_batch_only_neighbors'][dic_int_vars['numBatches']] = batch_occupancy_only_neighbor
    dic_all_occupancies['occupancies_per_batch_only_neighbors_only_edeps'][dic_int_vars['numBatches']] = batch_occupancy_only_neighbor_only_edep
    
    #given pos_only_neighbors, we will now calculate the mcID of the cells that were fired
    cell_to_mcID_neighbors = {}
    for i, key in enumerate(batchVars['pos_only_neighbors']):
        unique_layer_index = key[0]
        nphi = key[1]
        stereoSign = key[2]
        current_pos_full = (unique_layer_index, nphi, stereoSign)
        
        #find where in cell_to_mcID the unique_layer_index is
        # index = [i for i, tup in enumerate(cell_to_mcID) if tup[0] == unique_layer_index and tup[1] == nphi]
        if current_pos_full not in dic_posToKey_by_batch:
            print(f"Error: unique_layer_index not found in cell_to_mcID: {current_pos_full}")
            input("Press Enter to continue...")
        else:
            if current_pos_full not in cell_to_mcID_neighbors:
                cell_to_mcID_neighbors[current_pos_full] = dic_posToKey_by_batch[current_pos_full]['mcID_index'] #setup dictionary for current cell's neighbors
            else:
                if type(dic_posToKey_by_batch[current_pos_full]) is not list:
                    print(f"Warning: cell_to_mcID_neighbors is not a list: {dic_posToKey_by_batch[current_pos_full]}")
                    input("Press Enter to continue...")
                cell_to_mcID_neighbors[current_pos_full] += (dic_posToKey_by_batch[current_pos_full]['mcID_index']) #append the mcID's of the cell to the list 
    #same for pos_only_neighbors_only_edeps 
    cell_to_mcID_neighbors_only_edep = {}
    for i,key in enumerate(batchVars['pos_only_neighbors_only_edeps']):
        unique_layer_index = key[0]
        nphi = key[1]
        stereoSign = key[2]
        current_pos_full = (unique_layer_index, nphi, stereoSign)
        # print(f"unique_layer_index: {unique_layer_index}, nphi: {nphi}, energy: {energy}")
        #find where in cell_to_mcID the unique_layer_index is
        if current_pos_full not in dic_posToKey_by_batch:
            print(f"Error: unique_layer_index not found in cell_to_mcID: {current_pos_full}")
            input("Press Enter to continue...")
        else:
            # if dic_cell_to_mcID[(unique_layer_index, nphi)] in cell_to_mcID_neighbors_only_edep:
            #     print(f"\n \n \n Warning: duplicate key in cell_to_mcID_neighbors_only_edep: {unique_layer_index}, {nphi} \n \n \n")
            #     input("Press Enter to continue...")
            if current_pos_full not in cell_to_mcID_neighbors_only_edep:
                cell_to_mcID_neighbors_only_edep[current_pos_full] = dic_posToKey_by_batch[current_pos_full]
            else:
                cell_to_mcID_neighbors_only_edep[current_pos_full] += (dic_posToKey_by_batch[current_pos_full])

    return dic_int_vars, dic_posToKey_by_batch, dic_RPhiKey_by_batch, dic_list_mcIDs_one_batch, dic_all_occupancies, batchVars
    
def updateOcc(typeFile="bkg", numfiles=500, radiusR=1, radiusPhi=-1, atLeast=1, edepRange=0.05, edepAtLeast=1, edepLoosen=False, flexible=True, zrange=0):
    print("Calculating occupancy data from files...")
    list_overlay = []
    
    if radiusPhi == -1:
        radiusPhi = radiusR
        
    #setup dictionary
    dic = {}
    #can change dic_file_path to the correct path:
    dic_file_path = "/eos/user/a/alpoulin/fccBBTrackData/noOcc/" + str(typeFile) + "_background_particles_" + str(numfiles) + ".npy" #cernbox (to save storage)
    output_dic_file_path = "/eos/user/a/alpoulin/fccBBTrackData/rphiedepz/" + str(typeFile) + "_background_particles_" + str(numfiles)  + "_v6" + \
        "_R" + str(radiusR) + "_P" + str(radiusPhi) + "_AL" + str(atLeast) + "_ER" + str(edepRange) + "_EAL" + str(edepAtLeast) + "_EL" + str(int(edepLoosen)) + "_ZR" + str(zrange) + ".npy" #cernbox (to save storage)
    dic_keys = ["list_n_cells_fired_mc", "max_n_cell_per_layer",
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
        "neighborPt_by_batch", "neighborPDG_by_batch"] #currently outdated
    #check if dic_file_path exists: 
    try:
        dic = np.load(dic_file_path, allow_pickle=True).item()
        print(f"Dictionary loaded from {dic_file_path}")
        for key in dic_keys:
            dic[key] = []
    except:
        print(f"Dictionary not found at {dic_file_path}")
        input("Press Enter to continue...")
        print("Creating new dictionary")
        #assign dic to empty dictionary
        dic = {}
        for key in dic_keys:
            dic[key] = []
        np.save(dic_file_path, dic)
        
        
    # print(f"typeFile: {typeFile}")
    print(f"Output_dic_file_path once finished will be: {output_dic_file_path}")
    
    bkgDataPath, combinedDataPath, bkgFilePath, combinedFilePath, signalFilePath, signalDataPath = configure_paths(typeFile)

    list_overlay = setUpFiles(typeFile, flexible, numfiles, bkgDataPath, combinedDataPath, bkgFilePath, combinedFilePath, signalFilePath, signalDataPath)
    
    total_number_of_layers, n_cell_per_layer, n_cell_per_superlayer, max_n_cell_per_layer, total_number_of_cells, n_layers_per_superlayer, n_superlayers, list_max_n_cell_per_layer = driftChamberProperties()
    
    
    
    
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w') # Temporarily Suppress print statements since opening detector file takes a while
    DCHi = DCH_info(detectorPath = os.path.join(os.environ["K4GEO"], "FCCee/IDEA/compact/IDEA_o1_v03/IDEA_o1_v03.xml"))
    sys.stdout.close()
    sys.stdout = original_stdout # Re-enable print statements
    # print(DCHi.get_database_as_list_dic())
    z_layer_to_shift = layerDrift(DCHi, n_cell_per_layer)
    # print(f"z_layer_to_shift: {z_layer_to_shift}")
    # input("Press Enter to continue...")
    
    if typeFile=="bkg": #we want to get the occupancy for 20 events/files at a time
        batches=20
        eventFactor=1 #one event per file
    elif typeFile=="signal": #we want to get the occupancy for 1 event at a time
        batches=1
        eventFactor=10 #10 events per file
    elif typeFile=="combined": #we want to get the occupancy for 1 event at a time
        batches=1
        eventFactor=10 #10 events per file
        

    dic_all_occupancies = {}
    occupancies_keys = ["occupancies_per_batch_sum_batch", "occupancies_per_batch_only_neighbors", "occupancies_per_batch_only_neighbors_only_edeps", 
                        "avg_energy_dep_per_batch",
                        "occupancies_per_batch_only_bkg", "occupancies_per_batch_only_signal", 
                        "occupancies_per_batch_only_bkg_only_neighbors", "occupancies_per_batch_only_signal_only_neighbors",
                        "occupancies_per_batch_only_bkg_only_neighbors_only_edeps", "occupancies_per_batch_only_signal_only_neighbors_only_edeps",]
    for key in occupancies_keys:
        dic_all_occupancies[key] = np.zeros((int(eventFactor*numfiles/batches), total_number_of_layers)) #we want it to be (500/20, 14) so 14 across 25 down
    
    print(f"Number of batches: {dic_all_occupancies['occupancies_per_batch_sum_batch'].shape[0]} \n") #number of batches
    
    dic_occupancies_per_batch_sum_batch_energy_dep_per_cell = {} #this will be a dictionary of np arrays
    
    dic_posToKey_by_batch_keys = ["mcID_index", #"mcID_only_neighbors", "mcID_only_neighbors_only_edeps", #mcID's
                                     "cell_fired_pos", #"cell_fired_pos_only_neighbors", "cell_fired_pos_only_neighbors_only_edeps",
                                     #"cell_fired_pos_of_neighbors", #all positions
                                    #  "energy_dep_per_cell", #"energy_dep_per_cell_only_neighbors", 
                                    #  "energy_dep_per_cell_non_acc",
                                     #"energy_dep_per_cell_only_neighbors_only_edeps", #energy deposition
                                     "energy_dep_per_cell_RPhiZ",
                                     "energy_dep_per_cell_RPhiZ_noacc",
                                     "energy_dep_per_cell_xyz_noacc",
                                     "pT", "PDG", "prod_sec", "photon_par", "gen_status", #misc
                                     
                                     #below are the ones more designed for combined only
                                     "combined_overlay_status", 
                                    ]
    
    dic_xyz_by_batch_keys = ["energy_dep_per_cell_xyz_noacc", ] #this will be a dictionary of np arrays
    
    dic_RPhiKey_by_batch_keys = ["pos_full", "energy_dep_per_cell", "energy_dep_per_cell_non_acc", "energy_dep_per_cell_only_neighbors", "energy_dep_per_cell_only_neighbors_only_edeps"] 
    list_RPhiKey_by_batch = [] #this will be a list of dictionaries where each dictionary is a batch, each dictionary has the key tuple (unique_layer_index, nphi) to a dictionary of keys from dic_posToKey_by_batch_keys, the values of which is a list of values (since we can have multiple hits in the same cell)
        
    list_posToKey_by_batch = [] #this will be a list of dictionaries where each dictionary is a batch, each dictionary has the key tuple (unique_layer_index, nphi, z) to a dictionary of keys from dic_posToKey_by_batch_keys, the values of which is a list of values (since we can have multiple hits in the same cell)
    
    list_xyz_by_batch = [] #this will be a list of dictionaries where each dictionary is a batch, each dictionary has the key tuple (x, y, z) to a dictionary of keys from dic_xyz_by_batch_keys, the values of which is a list of values (since we can have multiple hits in the same cell)
    

    dic_list_mcIDs_by_batch_keys = ["mcID_index_all", 
                                    "mcID_index_only_neighbors", "mcID_index_only_neighbors_only_edeps",
                                    "mcID_index_bkg", "mcID_index_signal",
                                    "mcID_index_bkg_only_neighbors", "mcID_index_signal_only_neighbors","mcID_index_bkg_only_neighbors_only_edeps", "mcID_index_signal_only_neighbors_only_edeps"] #dictionary of lists
    dic_list_mcIDs_by_batch = [] #this will be a list of dic_list_mcIDs_by_batch_keys for each batch
        
    dic_int_vars = {}
    dic_int_vars_keys = ["numBatches", "NoNeighborsRemoved", "NeighborsRemained", "EdepNeighborsRemained", "NoEdepNeighborsRemoved", "list_max_n_cell_per_layer", "list_max_n_cell_per_superlayer", "n_cell_per_layer", "n_cell_per_superlayer", "z_layer_to_shift", "total_number_of_cells", "total_number_of_layers", "radiusR", "radiusPhi", "atLeast", "edepRange", "edepAtLeast", "edepLoosen", "zrange"]
    dic_int_vars["numBatches"] = 0
    dic_int_vars["NoNeighborsRemoved"] = 0
    dic_int_vars["NeighborsRemained"] = 0
    dic_int_vars["EdepNeighborsRemained"] = 0
    dic_int_vars["NoEdepNeighborsRemoved"] = 0
    dic_int_vars["list_max_n_cell_per_layer"] = list_max_n_cell_per_layer
    dic_int_vars['list_max_n_cell_per_superlayer'] = []
    for i in range(0, len(dic_int_vars["list_max_n_cell_per_layer"]), 8): #for each superlayer, we want to get the max number of cells per superlayer
        dic_int_vars['list_max_n_cell_per_superlayer'].append(dic_int_vars["list_max_n_cell_per_layer"][i])
    print(f"max_n_cell_per_superlayer: {dic_int_vars['list_max_n_cell_per_superlayer']}")
    print(f"list_max_n_cell_per_layer: {dic_int_vars['list_max_n_cell_per_layer']}")
    dic_int_vars["n_cell_per_layer"] = n_cell_per_layer
    dic_int_vars["n_cell_per_superlayer"] = n_cell_per_superlayer
    dic_int_vars["z_layer_to_shift"] = z_layer_to_shift
    dic_int_vars["total_number_of_cells"] = total_number_of_cells
    dic_int_vars["total_number_of_layers"] = total_number_of_layers
    dic_int_vars["radiusR"] = radiusR
    dic_int_vars["radiusPhi"] = radiusPhi
    dic_int_vars["atLeast"] = atLeast
    dic_int_vars["edepRange"] = edepRange
    dic_int_vars["edepAtLeast"] = edepAtLeast
    dic_int_vars["edepLoosen"] = edepLoosen
    dic_int_vars["zrange"] = zrange
    dic_int_vars["zpos"] = zsteps(DCHi)
    dic_int_vars["zStep"] = 100
    
    print(list_max_n_cell_per_layer)
    # input("Press Enter to continue...")
    
    NoNeighborsRemoved = 0
    NeighborsRemained = 0
    EdepNeighborsRemained = 0
    NoEdepNeighborsRemoved = 0
    


    # numBatches = 0 
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
        
        
        #reset the batch file mean after starting a new batch
        if i % batches == 0:
            if typeFile == "bkg": #want to reset every 20 bkg files
                batchVars = setupBatchVars(typeFile, dic_int_vars["numBatches"], 1)
                dic_posToKey_by_batch = {} #this will be a dictionary of key tuple: (unique_layer_index, nphi, stereoAngle) to a list of tuples (unique_layer_index, nphi, mcID) for each batch
                #note that values for any given key is a list of tuple, now most of the time it will be a single tuple, but in the case of multiple hits in the same exact cell, it will be len of the list > 1
                dic_RPhiKey_by_batch = {}
                dic_list_mcIDs_one_batch = {} #dic of mcID's for each batch; to make it easier to index their values
                dic_xyz_by_batch = {}
                for key in dic_list_mcIDs_by_batch_keys:
                                dic_list_mcIDs_one_batch[key] = []
        
        numEvents = 0
        for event in reader.get("events"):
            # print(f"Running over event: {numEvents}")
            numEvents += 1
            
            EventMCParticles = event.get("MCParticles")
            dc_hits = event.get("DCHCollection")
            
            
            if typeFile == "signal": #want to reset every 1 signal event
                batchVars = setupBatchVars(typeFile, dic_int_vars["numBatches"], 0)
                dic_posToKey_by_batch = {} #this will be a dictionary of key tuple: (unique_layer_index, nphi, stereoAngle) to a list of tuples (unique_layer_index, nphi, mcID) for each batch
                #note that values for any given key is a list of tuple, now most of the time it will be a single tuple, but in the case of multiple hits in the same exact cell, it will be len of the list > 1
                dic_RPhiKey_by_batch = {}
                dic_list_mcIDs_one_batch = {} #dic of mcID's for each batch; to make it easier to index their values
                dic_xyz_by_batch = {}
                for key in dic_list_mcIDs_by_batch_keys:
                    dic_list_mcIDs_one_batch[key] = []
                    
                for key in occupancies_keys:
                    dic_all_occupancies[key][dic_int_vars["numBatches"]] = np.zeros((total_number_of_layers))
                
            if typeFile == "combined": #want to reset every 1 combined event since for each file, 10 signal events, each with 20 bkg events respectively
                batchVars = setupBatchVars(typeFile, dic_int_vars["numBatches"], 0)
                dic_posToKey_by_batch = {} #this will be a dictionary of key tuple: (unique_layer_index, nphi, stereoAngle) to a list of tuples (unique_layer_index, nphi, mcID) for each batch
                #note that values for any given key is a list of tuple, now most of the time it will be a single tuple, but in the case of multiple hits in the same exact cell, it will be len of the list > 1
                dic_RPhiKey_by_batch = {}
                dic_list_mcIDs_one_batch = {} #dic of mcID's for each batch; to make it easier to index their values
                dic_xyz_by_batch = {}
                for key in dic_list_mcIDs_by_batch_keys:
                    dic_list_mcIDs_one_batch[key] = []
                dc_hits = event.get("NewCDCHHits")
                
            
            
            for num_hit, dc_hit in enumerate(dc_hits):
                # print(f"Running over hit: {num_hit}")
                mcParticleHit = dc_hit.getMCParticle()
                index_mc = mcParticleHit.getObjectID().index
                if typeFile=="combined":
                    mcParticle = mcParticleHit
                else:
                    mcParticle = EventMCParticles[int(index_mc)] #segfaults for combined, idk why
                
                isBkgOverlay = 1 if typeFile == "Bkg" else 0 #set to 0 if signal, combined overwrites
                if typeFile == "combined":
                    isBkgOverlay = dc_hit.isOverlay()
                    particleType = 1 if isBkgOverlay else 0
                    batchVars['particleType'] = particleType
                
                cellID = dc_hit.getCellID()
                superlayer = decoder.get(cellID, "superlayer")
                layer = decoder.get(cellID, "layer")
                nphi = decoder.get(cellID, "nphi")
                stereosign = decoder.get(cellID, "stereosign")
                hit_z = dc_hit.z()
                closest_zpos_index = np.argmin(np.abs(dic_int_vars["zpos"] - hit_z))
                approx_hit_z = dic_int_vars["zpos"][closest_zpos_index] #get the closest z value ish
                
                # print(decoder.fields()) #get all the decoder has to offer
                # print(decoder.fieldDescription())
                
                # define a unique layer index based on super layer and layer
                if layer >= n_layers_per_superlayer or superlayer >= n_superlayers:
                    print("Error: layer or super layer index out of range")
                    print(f"Layer: {layer} while max layer is {n_layers_per_superlayer - 1}. \
                        Superlayer: {superlayer} while max superlayer is {n_superlayers - 1}.")
                unique_layer_index = superlayer * n_layers_per_superlayer + layer #we will also use this as the radial index
                cellID_unique_identifier = "SL_" + str(superlayer)  + "_L_" + str(layer) + "_nphi_" + str(nphi) 
                
                
                current_cell_fired_position_tuple = (unique_layer_index, nphi, approx_hit_z)
                current_cell_fired_position_tuple_xyz = (dc_hit.getPosition().x, dc_hit.getPosition().y, dc_hit.getPosition().z)
                current_cell_fired_position_tuple_onlyRPhi = (unique_layer_index, nphi) #this is the key we will use to index the dictionary, we dont need the z value since we are only interested in the occupancy of the cell
                
                if current_cell_fired_position_tuple not in dic_posToKey_by_batch: #have not seen this cell yet
                    dic_posToKey_by_batch[current_cell_fired_position_tuple] = {} #setup dictionary for current cell's data
                    for key in dic_posToKey_by_batch_keys: #initialize all keys to empty lists
                        dic_posToKey_by_batch[current_cell_fired_position_tuple][key] = []
                if current_cell_fired_position_tuple_onlyRPhi not in dic_RPhiKey_by_batch: #have not seen this cell yet
                    dic_RPhiKey_by_batch[current_cell_fired_position_tuple_onlyRPhi] = {}
                    for key in dic_RPhiKey_by_batch_keys: #initialize all keys to empty lists
                        dic_RPhiKey_by_batch[current_cell_fired_position_tuple_onlyRPhi][key] = []
                        
                if current_cell_fired_position_tuple_xyz not in dic_xyz_by_batch: #have not seen this cell yet
                    dic_xyz_by_batch[current_cell_fired_position_tuple_xyz] = {}
                    for key in dic_xyz_by_batch_keys: #initialize all keys to empty lists
                        dic_xyz_by_batch[current_cell_fired_position_tuple_xyz][key] = []

                #else this key already exists, we simply append to the current list for the key
                                    
                #get mcIndex
                dic_list_mcIDs_one_batch["mcID_index_all"].append(index_mc)
                dic_posToKey_by_batch[current_cell_fired_position_tuple]['mcID_index'].append(index_mc)
                
                #get combined overlay status
                if typeFile == "combined" and isBkgOverlay:
                    dic_list_mcIDs_one_batch["mcID_bkg"].append(index_mc)
                elif typeFile == "combined" and not isBkgOverlay:
                    dic_list_mcIDs_one_batch["mcID_signal"].append(index_mc)
                
                #get cell fired position
                dic_posToKey_by_batch[current_cell_fired_position_tuple]['cell_fired_pos'].append((unique_layer_index, nphi, approx_hit_z, dc_hit.getPosition().x, dc_hit.getPosition().y, dc_hit.getPosition().z, mcParticle.getGeneratorStatus()))
                
                #get gen status
                dic_posToKey_by_batch[current_cell_fired_position_tuple]['gen_status'].append(mcParticle.getGeneratorStatus())
                
                #get photon parent
                has_photon_parent = 0
                for parent in mcParticleHit.getParents():
                    if parent.getPDG() == 22:
                        has_photon_parent = 1
                        break
                dic_posToKey_by_batch[current_cell_fired_position_tuple]['photon_par'].append(has_photon_parent)
                
                #get PDG
                dic_posToKey_by_batch[current_cell_fired_position_tuple]['PDG'].append(mcParticle.getPDG())
                
                #get pT
                pt = math.sqrt(mcParticle.getMomentum().x**2 + mcParticle.getMomentum().y**2)
                dic_posToKey_by_batch[current_cell_fired_position_tuple]['pT'].append(pt)
                
                #get production secondary
                dic_posToKey_by_batch[current_cell_fired_position_tuple]['prod_sec'].append(dc_hit.isProducedBySecondary())
                
                #combined overlay status
                dic_posToKey_by_batch[current_cell_fired_position_tuple]["combined_overlay_status"].append(isBkgOverlay)
                
                #get edep (r, phi, z)
                dic_posToKey_by_batch[current_cell_fired_position_tuple]['energy_dep_per_cell_RPhiZ_noacc'].append((dc_hit.getEDep()))
                
                #get edep (non accumulating)
                dic_RPhiKey_by_batch[current_cell_fired_position_tuple_onlyRPhi]['energy_dep_per_cell_non_acc'].append((dc_hit.getEDep()))
                
                #put into pos_full
                dic_RPhiKey_by_batch[current_cell_fired_position_tuple_onlyRPhi]['pos_full'].append((unique_layer_index, nphi, approx_hit_z))
                
                dic_xyz_by_batch[current_cell_fired_position_tuple_xyz]['energy_dep_per_cell_xyz_noacc'].append((dc_hit.getEDep()))
                        
                        
                # only phi,R
                if not cellID_unique_identifier in batchVars['dict_cellID_nHits'].keys(): # the cell was not fired yet
                    batchVars['dict_cellID_nHits'][cellID_unique_identifier] = 1
                    
                    batchVars['pos'].append((unique_layer_index, nphi))
                    
                    dic_RPhiKey_by_batch[current_cell_fired_position_tuple_onlyRPhi]['energy_dep_per_cell'].append((dc_hit.getEDep()))
                    
                    if typeFile=="combined" and isBkgOverlay:
                        batchVars['pos_bkg_overlay'].append((unique_layer_index, nphi))
                    elif typeFile=="combined" and not isBkgOverlay:
                        batchVars['pos_signal_overlay'].append((unique_layer_index, nphi))
                        
                    #put into pos_full
                    # dic_RPhiKey_by_batch[current_cell_fired_position_tuple_onlyRPhi]['pos_full'] = [(unique_layer_index, nphi, approx_hit_z)]
                        
                else: # the cell was already fired
                    batchVars['dict_cellID_nHits'][cellID_unique_identifier] += 1
                    
                    dic_RPhiKey_by_batch[current_cell_fired_position_tuple_onlyRPhi]['energy_dep_per_cell'][0] = (dic_RPhiKey_by_batch[current_cell_fired_position_tuple_onlyRPhi]['energy_dep_per_cell'][0] + dc_hit.getEDep()) #accumulate the edep

                    
                # phi,R,z
                if current_cell_fired_position_tuple not in batchVars['dict_cellID_nHits_full']:
                    batchVars['dict_cellID_nHits_full'][current_cell_fired_position_tuple] = 1
                    
                    dic_posToKey_by_batch[current_cell_fired_position_tuple]['energy_dep_per_cell_RPhiZ'].append(dc_hit.getEDep())
                    # print(f"rphiz pre: {dic_posToKey_by_batch[current_cell_fired_position_tuple]['energy_dep_per_cell_RPhiZ']} for {cellID_unique_identifier, current_cell_fired_position_tuple}")
                else:
                    batchVars['dict_cellID_nHits_full'][current_cell_fired_position_tuple] += 1
                    # print(f"rphiz post: {dic_posToKey_by_batch[current_cell_fired_position_tuple]['energy_dep_per_cell_RPhiZ']} for {cellID_unique_identifier, current_cell_fired_position_tuple}")
                    
                    dic_posToKey_by_batch[current_cell_fired_position_tuple]['energy_dep_per_cell_RPhiZ'][0] = (dic_posToKey_by_batch[current_cell_fired_position_tuple]['energy_dep_per_cell_RPhiZ'][0] + dc_hit.getEDep())
                    
            ###end of hit loop
            
            
            ##signal files we want to reset every event
            if typeFile == "signal":
                print(f"setting occupancy for batch: {dic_int_vars['numBatches']}")
                dic_int_vars, dic_posToKey_by_batch, dic_RPhiKey_by_batch, dic_list_mcIDs_one_batch, dic_all_occupancies, batchVars = calcOcc(dic_int_vars, dic_posToKey_by_batch, dic_RPhiKey_by_batch, dic_list_mcIDs_one_batch, dic_all_occupancies, batchVars) #
                
                
                list_posToKey_by_batch.append(dic_posToKey_by_batch)
                list_RPhiKey_by_batch.append(dic_RPhiKey_by_batch)
                dic_list_mcIDs_by_batch.append(dic_list_mcIDs_one_batch)
                list_xyz_by_batch.append(dic_xyz_by_batch)
                
                dic_int_vars["numBatches"] += 1
                
            if typeFile=="combined":
                print(f"setting occupancy for batch: {dic_int_vars['numBatches']}")
                batch_occupancy_only_bkg = []
                batch_occupancy_only_signal = []
                batch_occupancy_only_bkg_only_neighbor_only_edeps = []
                batch_occupancy_only_signal_only_neighbor_only_edeps = []
                for unique_layer_index in range(0, total_number_of_layers):
                    batch_occupancy_only_bkg.append(calculateOccupancy(batchVars["pos_bkg_overlay"], unique_layer_index, n_cell_per_layer))
                    batch_occupancy_only_signal.append(calculateOccupancy(batchVars["pos_bkg_overlay"], unique_layer_index, n_cell_per_layer))
                dic_all_occupancies['occupancies_per_batch_sum_batch_only_bkg'][dic_int_vars["numBatches"]] = batch_occupancy_only_bkg
                dic_all_occupancies['occupancies_per_batch_sum_batch_only_signal'][dic_int_vars["numBatches"]] = batch_occupancy_only_signal


                dic_int_vars, dic_posToKey_by_batch, dic_RPhiKey_by_batch, dic_list_mcIDs_one_batch, dic_all_occupancies, batchVars = calcOcc(dic_int_vars, dic_posToKey_by_batch, dic_RPhiKey_by_batch, dic_list_mcIDs_one_batch, dic_all_occupancies, batchVars)
                
                # cell_fired_pos_per_batch.append(batch_cell_fired_pos)
                # cell_fired_pos_neighbors_per_batch.append(batch_cell_fired_pos_neighbors)
                # energy_dep_per_cell_per_batch.append(occupancies_a_batch_edep) #a tuple of (unique_layer_index, nphi, edep)
                # energy_dep_per_cell_per_batch_only_neighbors.append(edep_only_neighbors)
                # energy_dep_per_cell_per_batch_only_neighbors_only_edeps.append(edep_only_neighbors_only_edep)
                # cell_to_mcID_per_batch.append(cell_to_mcID)
                # cell_to_mcID_neighbors_per_batch.append(cell_to_mcID_neighbors)
                # cell_to_mcID_neighbors_edeps_per_batch.append(cell_to_mcID_neighbors_edeps)
                # neighborPt.append(batch_pt)
                # list_dic_PDG.append(batch_pdg)
                # list_prod_sec.append(batch_prod_sec)
                # list_dic_photon_par.append(batch_photon_par)
                # combined_overlay_status.append(occupancies_a_batch_isBkgOverlay)
                # energy_dep_per_cell_per_batch_bkg.append(list(batch_edep_pos_bkg.values()))
                # energy_dep_per_cell_per_batch_signal.append(list(batch_edep_pos_signal.values()))
                # list_dic_cart_pos_status.append(batch_cart_pos_status)
                
                list_posToKey_by_batch.append(dic_posToKey_by_batch)
                list_RPhiKey_by_batch.append(dic_RPhiKey_by_batch)
                dic_list_mcIDs_by_batch.append(dic_list_mcIDs_one_batch)
                list_xyz_by_batch.append(dic_xyz_by_batch)

                dic_int_vars["numBatches"] += 1
        ###end of event loop
            
            
        #the next one resets the 20 file batch
        if (i + 1) % batches == 0 and typeFile == "bkg":
            print(f"setting occupancy for batch: {dic_int_vars['numBatches']}")
            dic_int_vars, dic_posToKey_by_batch, dic_RPhiKey_by_batch, dic_list_mcIDs_one_batch, dic_all_occupancies, batchVars = calcOcc(dic_int_vars, dic_posToKey_by_batch, dic_RPhiKey_by_batch, dic_list_mcIDs_one_batch, dic_all_occupancies, batchVars) #
                
                
            list_posToKey_by_batch.append(dic_posToKey_by_batch)
            list_RPhiKey_by_batch.append(dic_RPhiKey_by_batch)
            dic_list_mcIDs_by_batch.append(dic_list_mcIDs_one_batch)
            list_xyz_by_batch.append(dic_xyz_by_batch)
            
            dic_int_vars["numBatches"] += 1
            
            # cell_fired_pos_neighbors_per_batch.append(batch_cell_fired_pos_neighbors)
            # energy_dep_per_cell_per_batch_only_neighbors.append(edep_only_neighbors)
            # energy_dep_per_cell_per_batch_only_neighbors_only_edeps.append(edep_only_neighbors_only_edep)
            # cell_to_mcID_neighbors_per_batch.append(cell_to_mcID_neighbors)
            # cell_to_mcID_neighbors_edeps_per_batch.append(cell_to_mcID_neighbors_edeps)
            
            # list_posToKey_by_batch.append(dic_posToKey_by_batch)
            # list_RPhiKey_by_batch.append(dic_RPhiKey_by_batch)
            # dic_list_mcIDs_by_batch.append(dic_list_mcIDs_one_batch)
            # list_xyz_by_batch.append(dic_xyz_by_batch)
            
            # dic_int_vars["numBatches"] += 1
                    
                    
        # percentage_of_fired_cells.append(100 * len(dict_cellID_nHits.keys())/float(total_number_of_cells)  )
    ###end of file loop

    print("end of file loop")


    print("updating dictionary")
    dic['list_posToKey_by_batch'] = list_posToKey_by_batch
    dic['dic_list_mcIDs_by_batch'] = dic_list_mcIDs_by_batch
    dic['batchVars'] = batchVars
    dic['dic_int_vars'] = dic_int_vars
    
    
    #at this point we have filled the occupancy so there are 500 rows and 112 columns
    ddofFactor = 0
    dic["occupancy_per_batch_sum_batches"] = np.mean(dic_all_occupancies['occupancies_per_batch_sum_batch'], axis=0)
    dic["occupancy_per_batch_sum_batches_error"] = np.std(dic_all_occupancies['occupancies_per_batch_sum_batch'], axis=0, ddof=ddofFactor) / np.sqrt(dic_all_occupancies['occupancies_per_batch_sum_batch'].shape[0])
    #error is the std of the mean, i.e. std / sqrt(n)
    dic["occupancy_per_batch_sum_batches_non_meaned"] = dic_all_occupancies['occupancies_per_batch_sum_batch']

    dic["occupancy_per_batch_sum_batches_only_neighbor"] = np.mean(dic_all_occupancies['occupancies_per_batch_only_neighbors'], axis=0)
    dic["occupancy_per_batch_sum_batches_only_neighbor_error"] = np.std(dic_all_occupancies['occupancies_per_batch_only_neighbors'], axis=0, ddof=ddofFactor) / np.sqrt(dic_all_occupancies['occupancies_per_batch_only_neighbors'].shape[0])
    # print(f"no neighbors removed: {NoNeighborsRemoved}")
    # print(f"remained neighbors: {NeighborsRemained}")
    dic["no_neighbors_removed"] = dic_int_vars["NoNeighborsRemoved"]
    dic["neighbors_remained"] = dic_int_vars["NeighborsRemained"]
    dic["no_edep_neighbors_removed"] = dic_int_vars["NoEdepNeighborsRemoved"]
    dic["edep_neighbors_remained"] = dic_int_vars["EdepNeighborsRemained"]
    print(f"no neighbors removed: {dic['no_neighbors_removed']}")
    print(f"no edep neighbors removed: {dic['no_edep_neighbors_removed']}")
    
    dic["occupancy_per_batch_sum_batches_only_neighbor_only_edep"] = np.mean(dic_all_occupancies['occupancies_per_batch_only_neighbors_only_edeps'], axis=0)
    dic["occupancy_per_batch_sum_batches_only_neighbor_only_edep_error"] = np.std(dic_all_occupancies['occupancies_per_batch_only_neighbors_only_edeps'], axis=0, ddof=ddofFactor) / np.sqrt(dic_all_occupancies['occupancies_per_batch_only_neighbors_only_edeps'].shape[0])

    #given the dictionary of key to list of edep, we will now mean the list of edep
    dic["dic_occupancy_per_batch_sum_batches_energy_dep"] = {}
    for key in dic_occupancies_per_batch_sum_batch_energy_dep_per_cell.keys():
        dic["dic_occupancy_per_batch_sum_batches_energy_dep"][key] = np.mean(dic_occupancies_per_batch_sum_batch_energy_dep_per_cell[key])
        
    energy_dep_rphiz = []
    for batch in list_posToKey_by_batch:
        batch_rphiz = {}
        for pos_tuple in batch.keys():
            if len(batch[pos_tuple]['energy_dep_per_cell_RPhiZ']) > 1:
                RuntimeError("energy_dep_per_cell_RPhiZ is not a single value")
            else:
                batch_rphiz[pos_tuple] = batch[pos_tuple]['energy_dep_per_cell_RPhiZ']
        energy_dep_rphiz.append(batch_rphiz)
    # dic["energy_dep_per_cell_RPhiZ"] = [batch[pos_tuple]['energy_dep_per_cell_RPhiZ'] for batch in list_posToKey_by_batch for pos_tuple in batch.keys()]
    dic["energy_dep_per_cell_RPhiZ"] = energy_dep_rphiz #list of dictionaries; keys = pos, value = edep
    
    energy_dep_xyz = []
    for batch in list_xyz_by_batch:
        batch_xyz = {}
        for pos_tuple in batch.keys():
            if len(batch[pos_tuple]['energy_dep_per_cell_xyz_noacc']) > 1:
                RuntimeError("energy_dep_per_cell_xyz_noacc is not a single value")
            else:
                batch_xyz[pos_tuple] = batch[pos_tuple]['energy_dep_per_cell_xyz_noacc']
        energy_dep_xyz.append(batch_xyz)
    dic["energy_dep_per_cell_xyz_noacc"] = energy_dep_xyz #list of dictionaries; keys = pos, value = edep
    
    
    dic["n_cell_per_layer"] = n_cell_per_layer
    dic["n_cell_per_superlayer"] = n_cell_per_superlayer
    dic["total_number_of_cells"] = total_number_of_cells
    dic["total_number_of_layers"] = total_number_of_layers
    dic["max_n_cell_per_layer"] = max_n_cell_per_layer #this is max phi index!!!
    
    
    input("Press Enter to continue... temporary save")
    np.save(output_dic_file_path, dic)
    input("Press Enter to continue...")

    dic["energy_dep_per_cell_per_batch"] = energy_dep_per_cell_per_batch
    dic["energy_dep_per_cell_per_batch_only_neighbors"] = energy_dep_per_cell_per_batch_only_neighbors
    dic["energy_dep_per_cell_per_batch_only_neighbors_only_edeps"] = energy_dep_per_cell_per_batch_only_neighbors_only_edeps
    
    dic["energy_dep_per_cell_per_batch_bkg"] = energy_dep_per_cell_per_batch_bkg #really only combined
    dic["energy_dep_per_cell_per_batch_signal"] = energy_dep_per_cell_per_batch_signal #really only combined
    dic["energy_dep_per_cell_per_batch_bkg_only_neighbors_only_edeps"] = energy_dep_per_cell_per_batch_bkg_only_neighbors_only_edeps
    dic["energy_dep_per_cell_per_batch_signal_only_neighbors_only_edeps"] = energy_dep_per_cell_per_batch_signal_only_neighbors_only_edeps
    
    dic["occupancy_per_batch_sum_batches_only_bkg"] = np.mean(dic_all_occupancies['occupancies_per_batch_sum_batch_only_bkg'], axis=0)
    dic["occupancy_per_batch_sum_batches_only_bkg_error"] = np.std(dic_all_occupancies['occupancies_per_batch_sum_batch_only_bkg'], axis=0, ddof=ddofFactor) / np.sqrt(dic_all_occupancies['occupancies_per_batch_sum_batch_only_bkg'].shape[0])
    dic["occupancy_per_batch_sum_batches_only_signal"] = np.mean(dic_all_occupancies['occupancies_per_batch_sum_batch_only_signal'], axis=0)
    dic["occupancy_per_batch_sum_batches_only_signal_error"] = np.std(dic_all_occupancies['occupancies_per_batch_sum_batch_only_signal'], axis=0, ddof=ddofFactor) / np.sqrt(dic_all_occupancies['occupancies_per_batch_sum_batch_only_signal'].shape[0])
    
    dic["occupancies_per_batch_sum_batch_only_bkg_only_neighbor_only_edeps"] = np.mean(dic_all_occupancies['occupancies_per_batch_sum_batch_only_bkg_only_neighbors_only_edeps'], axis=0)
    dic["occupancies_per_batch_sum_batch_only_bkg_only_neighbor_only_edeps_error"] = np.std(dic_all_occupancies['occupancies_per_batch_sum_batch_only_bkg_only_neighbors_only_edeps'], axis=0, ddof=ddofFactor) / np.sqrt(dic_all_occupancies['occupancies_per_batch_sum_batch_only_bkg_only_neighbors_only_edeps'].shape[0])
    dic["occupancies_per_batch_sum_batch_only_signal_only_neighbor_only_edeps"] = np.mean(dic_all_occupancies['occupancies_per_batch_sum_batch_only_signal_only_neighbors_only_edeps'], axis=0)
    dic["occupancies_per_batch_sum_batch_only_signal_only_neighbor_only_edeps_error"] = np.std(dic_all_occupancies['occupancies_per_batch_sum_batch_only_signal_only_neighbors_only_edeps'], axis=0, ddof=ddofFactor) / np.sqrt(dic_all_occupancies['occupancies_per_batch_sum_batch_only_signal_only_neighbors_only_edeps'].shape[0])
    
    dic["combined_overlay_status_pos"] = combined_overlay_status #really only combined
    if typeFile == "combined":
        combinedDic = {}
        combinedDic["energy_dep_per_cell_per_batch"] = energy_dep_per_cell_per_batch
        combinedDic["energy_dep_per_cell_per_batch_bkg"] = energy_dep_per_cell_per_batch_bkg
        combinedDic["energy_dep_per_cell_per_batch_signal"] = energy_dep_per_cell_per_batch_signal
        #save as separate npy
        output = "public/work/fccproject-tracking/detector_beam_backgrounds/tracking/images/lxplus/combinedDicEdepPos"
        np.save(output, combinedDic)

    dic["occupancy_per_batch_sum_batch_avg_energy_dep"] = np.mean(occupancies_per_batch_sum_batch_energy_dep, axis=0)
    dic["occupancy_per_batch_sum_batch_avg_energy_dep_error"] = np.std(occupancies_per_batch_sum_batch_energy_dep, axis=0, ddof=ddofFactor) / np.sqrt(occupancies_per_batch_sum_batch_energy_dep.shape[0])
    dic["cell_fired_pos"] = cellFiredPos
    dic["cell_fired_pos_neighbors"] = cellFiredPosNeighbors
    dic["cell_fired_pos_by_batch"] = cell_fired_pos_per_batch
    
    dic["onlyNeighborMCID_per_batch"] = cell_to_mcID_neighbors_per_batch
    dic["onlyNeighborOnlyEdepMCID_per_batch"] = cell_to_mcID_neighbors_edeps_per_batch
    dic["cellFiredMCID_per_batch"] = cell_to_mcID_per_batch
    
    dic["neighborPt_by_batch"] = neighborPt
    dic["list_dic_PDG_by_batch"] = list_dic_PDG
    dic["list_dic_photon_par_by_batch"] = list_dic_photon_par
    dic["list_prod_sec_by_batch"] = list_prod_sec
    
    dic["list_dic_cart_pos_status_by_batch"] = list_dic_cart_pos_status



    


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
                        "\n-- edepLoosen(bool): Default(False)" +
                        "\n-- zrange(int): Default(0)",
                        type=str, default="", nargs='+')
    args = parser.parse_args()
    
    starttime = time.time()

    if args.calc and args.calc != "":
        # try:
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
            boolArg = True if args.calc[7] == "True" else False
            updateOcc(args.calc[0], int(args.calc[1]), int(args.calc[2]), int(args.calc[3]), int(args.calc[4]), float(args.calc[5]), int(args.calc[6]), boolArg)
        elif args.calc[0] in typeFile and len(args.calc) == 9:
            boolArg = True if args.calc[7] == "True" else False
            updateOcc(args.calc[0], int(args.calc[1]), int(args.calc[2]), int(args.calc[3]), int(args.calc[4]), float(args.calc[5]), int(args.calc[6]), boolArg, int(args.calc[8]))
        else:
            parser.error("Invalid fileType")
        # except ValueError as e:
        #     parser.error(str(e))
            
    endtime = time.time()
    print("Time taken: ", endtime - starttime)
    #'''