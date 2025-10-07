#Alexander Poulin Jan 2025
from podio import root_io
import numpy as np 
import math
import dd4hep as dd4hepModule
from ROOT import dd4hep
import sys
from trBkgDat import configure_paths, setUpFiles
from scipy.spatial import cKDTree
import argparse
import time
import os
from pyDCH_info import DCH_info
from utilities.utils import check_odd_fractions, globalPhiIndex, find_closest_indices, kappa, find_closest_point_after_redistribution, fast_check_odd_fractions, faster_check_odd_fractions

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
                       "pos_global_only_neighbors",
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
    batchVars["pos_global_only_neighbors"] = [] #a list of tuples (unique_layer_index, nphi, global_rphiz_pos)
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
    # print(f"n_cell_per_layer: {n_cell_per_layer}, list_max_n_cell_per_layer: {list_max_n_cell_per_layer}")
    
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

def zsteps(DCHi, steps = 100):
    """Generates the z steps for the drift chamber. This is done by looping over the z positions and calculating the t_r and theta for each layer. Currently set to 100 steps length.

    Args:
        None

    Returns:
        dict: dictionary which maps z position to a list of tuples (layer, t_r, theta) for each layer
    """
    
    stop = int(DCHi.lhalf)
    start = -int(DCHi.lhalf)
    step = steps
    zpos = np.arange(start, stop + step, step)  # Ensure stop is inclusive

    return zpos

def layerDrift(DCHi, n_cell_per_layer):
    """Generates the phi drift for a given layer and z position. This is done by looping over the z positions and calculating the t_r and theta for each layer.

    Args:
        None

    Returns:
        dict: dictionary which maps z position to a list of tuples (layer, radius, (xnew, ynew, z), arc_length_shift, arc_length_step) for each layer
    """
    
    phi_d = DCHi.twist_angle * np.pi / 180 #twist angle
    L = DCHi.lhalf * 2 #half length * 2
    k = kappa(L, phi_d)
    zpos = zsteps(DCHi)
    drift_chamber_info = {}
    radius_to_layer = {}

    z_layer_to_shift = {}
    for i, z in enumerate(zpos):
        layer_to_shift = []
        
        for j in range(0, len(DCHi._database)):
            #check layer
            if DCHi._database[j]['layer'] - 1 != j:
                print(f"WARNING... ... ... layer: {DCHi._database[j]['layer'] - 1} does not match index: {j} in database")
                continue
            else:
                if z == 0:
                    drift_chamber_info[j] = (DCHi._database[j]['nwires'], DCHi._database[j]['radius_sw_z0'])
                    radius_to_layer[int(DCHi._database[j]['radius_sw_z0'])] = j #map radius to layer index for z=0
            r_j = DCHi._database[j]['radius_sw_z0'] #radius of the wire at z=0
            stereoSign = DCHi._database[j]['stereo_sign']
            t_r, theta, arc_length_shift = stereoWire(r_j, z, 0, k*stereoSign, phi_d, L) #same for all layers
            
            arc_length_step = 2 * np.pi * r_j / n_cell_per_layer[str(j)] #arc length step for the layer
            layer_to_shift.append((j, r_j, t_r, arc_length_shift * stereoSign, arc_length_step))
            ### (layer, radius, (x_new, y_new, z), arc_length_shift, arc_length_step) ###
            if arc_length_step == 0:
                print(f"WARNING... ... ... arc_length_step: {arc_length_step} for layer: {j}, z: {z}, r_j: {r_j}, shifted_phi: {arc_length_shift * stereoSign}")
                input("Press Enter to continue... arc_length_step \n")
        z_layer_to_shift[z] = layer_to_shift
        # print(drift_chamber_info) if z == 0 else None #print only once for first z position
        # input("Press Enter to continue... z_layer_to_shift \n") if z == 0 else None #print only once for first z position
        print(f"radius_to_layer: {radius_to_layer}") if z == 0 else None #print only once for first z position
        
    return z_layer_to_shift, drift_chamber_info, radius_to_layer

def globalIndicieShift(z_layer_to_shift):
    """Generates the global indicies for a given layer and z position. This is done by looping over the z positions and calculating the t_r and theta for each layer.

    Args:
        None

    Returns:
        r_shiftedphi_z: dictionary key (layer, z) and value is the indicie of the shift
    """
    
    # print(f"z_layer_to_shift: {z_layer_to_shift}")
    # print(f"n_cell_per_layer: {n_cell_per_layer}")
    r_shiftedphi_z = {}
    for i, z in enumerate(z_layer_to_shift):
        for layer_i in range(0, len(z_layer_to_shift[z])):
            layer = z_layer_to_shift[z][layer_i][0] #layer indicie
            p_c = z_layer_to_shift[z][layer_i][1] #radius of the wire at z=0
            stereoSign = z_layer_to_shift[z][layer_i][3]
            # arc_length_step = 2 * np.pi * z / n_cell_per_layer[str(layer)] #arc length step for the layer
            arc_length_step = z_layer_to_shift[z][layer_i][4] #arc length step for the layer
            shifted_phi = z_layer_to_shift[z][layer_i][3] #this is the shift in phi for the layer
            # print(f"layer: {layer}, z: {z}, shifted_phi: {shifted_phi}, arc_length_step: {arc_length_step}")
            if arc_length_step < 0:
                print(f"WARNING... ... ... arc_length_step: {arc_length_step} for layer: {layer}, z: {z}, shifted_phi: {shifted_phi}")
                input("Press Enter to continue... arc_length_step \n")
            # print(f"globalIndicieShift: layer: {layer}, z: {z}, shifted_phi: {shifted_phi}, arc_length_step: {arc_length_step}")
            _, shifted_indicie = faster_check_odd_fractions(shifted_phi, arc_length_step, offset=0)
            # print(f"shifted_indicie: {shifted_indicie}")
            # shifted_indicie = shifted_indicie[1] if shifted_indicie[0] else 0
            
            r_shiftedphi_z[(layer, z)] = int(shifted_indicie / 2) #divide by 2 since only looking at 1/4Sp, int so it rounds towards 0
            
    #save numpy
    np.save("r_shiftedphi_z.npy", r_shiftedphi_z)
            
    return r_shiftedphi_z

def calcShiftIndicie(parent_zlayer_info, neighbor_zlayer_candidate_info, n_cell_per_layer, verbose = False):
    parent_stereo_sign = 1 if parent_zlayer_info[3] > 0 else 0 if parent_zlayer_info[3] < 0 else -1 #stereo sign of the parent layer; -1 if no stereo sign
    parent_layer_arc_length_step = parent_zlayer_info[4] #arc length step for the layer
    parent_layer = parent_zlayer_info[0] #layer index
    parent_superlayer = parent_layer // 8 #indexes at 0
    parent_phi_shift = parent_zlayer_info[3] #this is the shift in phi for the layer
    
    #need to check if each neighbor_zlayer shift and parent_zlayer shift are >= 1/4 layer phi step respectively; since radially swepted, both will cross their respective 1/4 phi step at the same time despite the different z positions and radial out
    
    
    
    ### see if there should be a shift in indicies
    neighbor_zlayer_candidate_stereo_sign = 1 if neighbor_zlayer_candidate_info[3] > 0 else 0 if neighbor_zlayer_candidate_info[3] < 0 else -1 #stereo sign of the neighbor layer; -1 if no stereo sign
    neighbor_layer = neighbor_zlayer_candidate_info[0] #layer index
    neighbor_candidate_superlayer = neighbor_layer // 8 #indexes at 0
    neighbor_layer_arc_length_step = neighbor_zlayer_candidate_info[4] #arc length step for the layer
    neighbor_zlayer_shift = neighbor_zlayer_candidate_info[3]
    
    parent_search_outwards = True if parent_layer < neighbor_layer else False #if the parent layer is less than the neighbor layer, we are searching outwards, otherwise we are searching backwards

    
    #basic assumption is that if the layer is the same, we can assume the shift is 0 since they are the same layer and will be shifting in parallel; same when if stereo sign is the same
    # if neighbor_layer == parent_layer or (neighbor_zlayer_candidate_stereo_sign == parent_stereo_sign and (parent_phi_shift != 0 and neighbor_zlayer_shift != 0)): #if the layer is the same, we can assume the shift is 0 since they are the same layer and will be shifting in parallel, same when if stereo sign is the same
    #     print(f"setting to 0 where before neighbor_zlayer_candidate_stereo_sign: {neighbor_zlayer_candidate_stereo_sign}, parent_stereo_sign: {parent_stereo_sign}, parent_phi_shift: {parent_phi_shift}, neighbor_zlayer_shift: {neighbor_zlayer_shift}") if verbose else None
    #     neighbor_zlayer_shift_indicie = 0 #i.e. if the stereo sign is the same, we can assume the shift is 0 since they are the same layer and will be shifting in parallel
    #     return neighbor_zlayer_shift_indicie #no need to check the shift if the layer is the same
    
    
    #this assumption should hold for the same superlayer
    print(f"calcShiftIndicie, parent: layer: {parent_layer}, shifted_phi: {parent_phi_shift}, arc_length_step: {parent_layer_arc_length_step},  parent_info: {parent_zlayer_info}") if verbose else None
    _, parent_sp_factor  = faster_check_odd_fractions(parent_phi_shift,parent_layer_arc_length_step) if parent_layer < neighbor_layer else faster_check_odd_fractions(parent_phi_shift, neighbor_layer_arc_length_step) #check if the shift is greater than 1/4 layer phi step #parent_step if neighbor_layer is greater than parent_layer since parent looking outwards, otherwise neighbor_step since ...
    
    # parent_phi_indicie_shift = check_parent_shift_indicie[1] if check_parent_shift_indicie[0] else 0 #this is the indicie of the shift #Todo, this is redundant since if false it [1] is already set to 0
    
    superlayer_sp_ratio = 1
    # print(f"neighbor_zlayer_candidate_stereo_sign: {neighbor_zlayer_candidate_stereo_sign}, parent_stereo_sign: {parent_stereo_sign}") if neighbor_zlayer_candidate_stereo_sign != parent_stereo_sign else None
    if neighbor_candidate_superlayer != parent_superlayer and neighbor_layer_arc_length_step != 0: #if the superlayer is different, we need to factor in the extra shift and re designate the index
        # closest_neighbor_candidate_index, new_neighbor_shift_from_original = find_closest_point_after_redistribution(n_cell_per_layer[parent_zlayer_info[0]], n_cell_per_layer[neighbor_zlayer_candidate_info[0]], added_points=48) #this will give us the closest point in the new division
        print(f"neighbor_layer_arc_length_step: {neighbor_layer_arc_length_step}, parent_layer_arc_length_step: {parent_layer_arc_length_step}, parent_search_outwards: {parent_search_outwards}, neighbor_info: {neighbor_zlayer_candidate_info}, parent_info: {parent_zlayer_info}") if verbose else None
        superlayer_sp_ratio = parent_layer_arc_length_step / neighbor_layer_arc_length_step #if parent_search_outwards else neighbor_layer_arc_length_step / parent_layer_arc_length_step
        #uses the assumption that the greater the layer, the smaller the arc length step
        ###Todo make sure this is correct order or if dependent on parent search out vs backwards
    
    if (neighbor_zlayer_candidate_stereo_sign != parent_stereo_sign) or (neighbor_candidate_superlayer != parent_superlayer): #check if the stereo sign is the same; if its the same, we can assume the shift is 0 since they will be shifting in parallel; this does not hold for across superlayers;
        # new_neighbor_shift_from_original = 0 #initialize to 0
        
        #arc length shift for the layer
        print(f"neighbor_zlayer_shift: {neighbor_zlayer_shift}, neighbor_layer_arc_length_step: {neighbor_layer_arc_length_step}, parent_layer_shift: {parent_zlayer_info[3]}, superlayer_sp_ratio: {superlayer_sp_ratio}") if verbose else None
        print(f"calcShiftIndicie neighbor: layer: {neighbor_layer}, shifted_phi: {neighbor_zlayer_shift}, arc_length_step: {neighbor_layer_arc_length_step}, neighbor_info: {neighbor_zlayer_candidate_info}, parent_info: {parent_zlayer_info}") if verbose else None
        _, neighbor_candidate_sp_factor = faster_check_odd_fractions(neighbor_zlayer_shift, neighbor_layer_arc_length_step) if parent_layer < neighbor_layer else faster_check_odd_fractions(neighbor_zlayer_shift, parent_layer_arc_length_step)#check if the shift is greater than 1/4 layer phi step
        
        # neighbor_zlayer_shift_indicie = check_neighbor_zlayer_shift_indicie[1] if check_neighbor_zlayer_shift_indicie[0] else 0 #this is the indicie of the shift; defaults to -1 if shift is not significant enough (over 1/4 layer phi step)
    else: #base conditions arent met so we can assume the shift is 0
        print(f"setting to 0 where before neighbor_zlayer_candidate_stereo_sign: {neighbor_zlayer_candidate_stereo_sign}, parent_stereo_sign: {parent_stereo_sign}, parent_phi_shift: {parent_phi_shift}, neighbor_zlayer_shift: {neighbor_zlayer_shift}, neighbor_candidate_superlayer: {neighbor_candidate_superlayer}, parent_superlayer: {parent_superlayer}") if verbose else None
        
        neighbor_zlayer_shift_indicie, parent_phi_indicie_shift = 0, 0 #i.e. if the stereo sign is the same, we can assume the shift is 0 since they are the same layer and will be shifting in parallel
        return neighbor_zlayer_shift_indicie #no need to check the shift if the layer is the same
    
    # print(f"parent_zlayer_info: {parent_zlayer_info}, neighbor_zlayer_candidate_info: {neighbor_zlayer_candidate_info}, parent_phi_indicie_shift: {parent_phi_indicie_shift}, neighbor_zlayer_shift_indicie: {neighbor_zlayer_shift_indicie}") if verbose else None
        
    # if neighbor_zlayer_shift_indicie != parent_phi_indicie_shift * -1 and neighbor_candidate_superlayer == parent_superlayer and verbose: #if the stereo sign is the same, we can assume the shift is 0 since they are the same layer and will be shifting in parallel
    #     print(f"WARNING... ... ... neighbor_zlayer_shift_indicie: {neighbor_zlayer_shift_indicie}, parent_phi_indicie_shift: {parent_phi_indicie_shift}; for layer: {parent_zlayer_info[0]}, superlayer: {parent_superlayer} shift: {parent_zlayer_info[3]}, dz: {parent_zlayer_info[2]}; for neighbor layer: {neighbor_zlayer_candidate_info[0]}, superlayer: {neighbor_candidate_superlayer}, shift: {neighbor_zlayer_shift},, dz: {neighbor_zlayer_candidate_info[2]}") #currently mostly coming up for backwards check across z
    #     # input("Press Enter to continue... neighbor_zlayer_shift_indicie \n")
    
    #should be able to assume at this point, parent and neighbor are different layers and either different superlayers or different stereo signs
    
    #condition check sp factors are opposite in sign except 0's
    if (neighbor_candidate_sp_factor != 0 and parent_sp_factor != 0) and np.sign(neighbor_candidate_sp_factor) != np.sign(parent_sp_factor):
        print(f"WARNING... ... ... neighbor_candidate_sp_factor: {neighbor_candidate_sp_factor}, parent_sp_factor: {parent_sp_factor}; for layer: {parent_zlayer_info[0]}, superlayer: {parent_superlayer} shift: {parent_zlayer_info[3]}, dz: {parent_zlayer_info[2]}; for neighbor layer: {neighbor_zlayer_candidate_info[0]}, superlayer: {neighbor_candidate_superlayer}, shift: {neighbor_zlayer_shift}, dz: {neighbor_zlayer_candidate_info[2]}") if verbose else None
        
    print(f"parent_sp_factor: {parent_sp_factor}, neighbor_candidate_sp_factor: {neighbor_candidate_sp_factor}") if verbose else None
    
    print(f"sp_factor_sum: {neighbor_candidate_sp_factor - parent_sp_factor * superlayer_sp_ratio}, neighbor_candidate_sp_factor: {neighbor_candidate_sp_factor}, parent_sp_factor: {parent_sp_factor}, superlayer_sp_ratio: {superlayer_sp_ratio}") if verbose else None
    
    neighbor_candidate_shift_indicie = int((neighbor_candidate_sp_factor - (parent_sp_factor * superlayer_sp_ratio)) / 2) * -1
    # if neighbor_candidate_sp_factor < 0:
    #     neighbor_candidate_shift_indicie = int((neighbor_candidate_sp_factor - parent_sp_factor) / 2)
    # else:
    #     neighbor_candidate_shift_indicie = int((neighbor_candidate_sp_factor + parent_sp_factor) / 2)
    
    return neighbor_candidate_shift_indicie
        
    # return neighbor_zlayer_shift_indicie if abs(neighbor_zlayer_shift_indicie) >= abs(parent_phi_indicie_shift) or neighbor_zlayer_shift_indicie == parent_phi_indicie_shift * -1 else parent_phi_indicie_shift #for now
     
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
    
def calculateOnlyNeighbors(dic_int_vars, dic_posToKey_by_batch, dic_RPhiKey_by_batch, dic_list_mcIDs_one_batch, dic_all_occupancies, batchVars, maxLayer=112, verbose=False):
    #calculate the occupancy of non-neighbor cells
    #we will loop over all the cells and check if they have neighbors
    #a neightbor will be defined if there exists an occupancy index where (unique_layer_index +-0 or 1, nphi +- 0 or 1) exists
    #if they do, we will remove them from the list
    #occupancy is a list of tuples (unique_layer_index, nphi)
    #we will return a list of unique_layer_index
    
    radiusR, radiusPhi, atLeast, edepRange, edepAtLeast, edepLoosen, zrange = dic_int_vars['radiusR'], dic_int_vars['radiusPhi'], dic_int_vars['atLeast'], dic_int_vars['edepRange'], dic_int_vars['edepAtLeast'], dic_int_vars['edepLoosen'], dic_int_vars['zrange']
    
    NoNeighborsRemoved, NeighborsRemained, EdepNeighborsRemained, NoEdepNeighborsRemoved = dic_int_vars['NoNeighborsRemoved'], dic_int_vars['NeighborsRemained'], dic_int_vars['EdepNeighborsRemained'], dic_int_vars['NoEdepNeighborsRemoved']
    
    maxnphiPerSuperLayer, maxnphiPerLayer = dic_int_vars['list_max_n_cell_per_superlayer'], dic_int_vars['list_max_n_cell_per_layer']
    
    z_layer_to_shift = dic_int_vars['z_layer_to_shift']
    zstepFactor = dic_int_vars["zStep"]
    maxLayerIndex = maxLayer - 1
    
    # Build once
    # print([dic_posToKey_by_batch[pos]['global_rphiz_pos'][0] for pos in dic_posToKey_by_batch.keys()]) if verbose else None#note currently not unique so length is not always 1
    
    global_rphiz_points = np.array([tuple(dic_posToKey_by_batch[pos]['global_rphiz_pos'][0]) for pos in batchVars['pos']])
    global_tree = cKDTree(global_rphiz_points)
    
    dicNeighbors = {} #will be a dictionary where key is pos of some cell fired, the value will be a list of neighbor pos
    dicGlobalNeighbors = {} #will be a dictionary where key is pos of some cell fired, the value will be a list of neighbor pos in global coordinates
    dicEdepNeighbors = {} #setup dictionary for current cell's neighbors edep
    
    # for i, key in enumerate(list(dic_posToKey_by_batch.keys())):
    for i, key in enumerate(batchVars['pos']): #where i is the layer number ##Todo are we assuming the keys are unique? i.e. no duplicates?
        unique_layer_index = key[0]
        superLayerIndex = unique_layer_index // 8 #indexes at 0
        nphi = key[1]
        hit_z = key[2] #already defined in closest z to zpos
        
        n_cells_in_layer = maxnphiPerSuperLayer[superLayerIndex] #number of cells in the layer
 
        parent_shift_info = z_layer_to_shift[hit_z][unique_layer_index] #get the shift info for the layer
        
        # current_pos = (unique_layer_index, nphi)
        current_pos_full = (unique_layer_index, nphi, hit_z)
        current_global_pos_full = dic_posToKey_by_batch[current_pos_full]['global_rphiz_pos'][0] #note still not unique so length is not always 1
        # print(f"global_pos_full: {current_global_pos_full}") if current_global_pos_full[0] > 2000 or current_global_pos_full[1] > 1 or current_global_pos_full[2] > 1000 else None
    
        currentEdep = dic_posToKey_by_batch[current_pos_full]['energy_dep_per_cell_RPhiZ'][0] if len(dic_posToKey_by_batch[current_pos_full]['energy_dep_per_cell_RPhiZ']) == 1 else RuntimeError("check since more than one edep") #energy deposition in the cell if we take edep for entire wire
        print(f"Checking for neighbors for parent: {current_pos_full}") if verbose else None
        
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
        
        neighbors = False
        neighborsEdep = False
        if current_pos_full not in dicNeighbors: #setup current cell if not seen before
            dicNeighbors[current_pos_full] = [] #setup dictionary for current cell's neighbors
        if current_pos_full not in dicEdepNeighbors:
            dicEdepNeighbors[current_pos_full] = [] #setup dictionary for current cell's neighbors edep
        if current_global_pos_full not in dicGlobalNeighbors: #setup current cell if not seen before
            dicGlobalNeighbors[current_global_pos_full] = [] #setup dictionary for current cell's neighbors in global coordinates
            
            
        #based on the z and r, we can determine what phi indicies to look at
        #for now we will look at global ranges, but we can also look at relative ranges/indicies later
        for dradius in range(-rangeR, rangeR + 1):
            numNeighborsFound = len(dicNeighbors[current_pos_full])
            numEdepNeighborsFound = len(dicEdepNeighbors[current_pos_full])
            numGlobalNeighborsFound = len(dicGlobalNeighbors[current_global_pos_full])
            
            if numNeighborsFound >= neighborAtLeast and numEdepNeighborsFound >= edepNeighborAtLeast and numGlobalNeighborsFound >= neighborAtLeast: #have we already seen enough
                print(f"Already seen enough neighbors for {current_pos_full}, skipping further checks") if verbose else None
                neighbors = True
                neighborsEdep = True
                break #skip if we have already seen enough neighbors and break out of dx loop
            
            cyclic_unique_layer_index = unique_layer_index + dradius #this is the layer index we are looking at #should not be cyclic
            neighborSuperlayer = (cyclic_unique_layer_index) // 8 #indexes at 0
            
            if dradius == 0 or cyclic_unique_layer_index < 0 or cyclic_unique_layer_index >= maxLayer: #check boundaries
                print(f"Skipping dradius: {dradius} for unique_layer_index: {unique_layer_index}, cyclic_unique_layer_index: {cyclic_unique_layer_index}") if verbose else None
                continue
            
            # print(f"rangez: {rangeZ}")
            for dz in range(-rangeZ, rangeZ + 1, dic_int_vars['zStep']):
                # print(f"Checking dz: {dz}") if verbose else None
                #now we want to determine possible neighbors in the phi direction; but radius and z should remain like normal
                if dz == 0:
                    print(f"Skipping dz: {dz} for unique_layer_index: {unique_layer_index}, cyclic_unique_layer_index: {cyclic_unique_layer_index}") if verbose else None
                    continue
                noncyclic_z = hit_z + dz
                if abs(noncyclic_z) >= len(z_layer_to_shift) / 2 * zstepFactor: #check boundaries
                    print(f"Skipping noncyclic_z: {noncyclic_z} since zlayer greater than: {len(z_layer_to_shift), len(z_layer_to_shift) / 2 * zstepFactor}, for unique_layer_index: {unique_layer_index}, cyclic_unique_layer_index: {cyclic_unique_layer_index}") if verbose else None
                    input("Press Enter to continue... noncyclic_z \n") if verbose and abs(noncyclic_z) < 2000 else None
                    continue
                
                print(f"noncyclic_z: {noncyclic_z}, unique_layer_index: {unique_layer_index}, cyclic_unique_layer_index: {cyclic_unique_layer_index}, radiusR: {rangeR}, radiusPhi: {radiusPhi}, rangeZ: {rangeZ}") if verbose else None
                
                if numNeighborsFound >= neighborAtLeast and numEdepNeighborsFound >= edepNeighborAtLeast: #if we have  already seen enough neighbors
                    print(f"Already seen enough neighbors for {current_pos_full}, skipping further checks") if verbose else None
                else:
                    neighbor_zlayer_candidate_info = z_layer_to_shift[noncyclic_z][min(cyclic_unique_layer_index, maxLayerIndex)] #get the shift info for the layer
                    
                    
                    shift_indicie = calcShiftIndicie(parent_shift_info, neighbor_zlayer_candidate_info, list(dic_int_vars['n_cell_per_layer'].values()))
                    print(f"shift_indicie: {shift_indicie}, parent_shift_info: {parent_shift_info}, neighbor_zlayer_candidate_info: {neighbor_zlayer_candidate_info}") if verbose else None
                
                for dphi in range(-rangePhi, rangePhi + 1):
                    if dphi == 0 and dradius == 0 and dz == 0: # Skip the center point
                        print(f"Skipping center point for unique_layer_index: {unique_layer_index}, nphi: {nphi}, dradius: {dradius}, dphi: {dphi}, dz: {dz}") if verbose else None
                        continue
                    
                    if numNeighborsFound >= neighborAtLeast and numEdepNeighborsFound >= edepNeighborAtLeast: #if we have already seen enough neighbors
                        print(f"Already seen enough neighbors for {current_pos_full}, skipping further checks") if verbose else None
                    else:
                        cyclic_nphi = ((nphi + dphi) + shift_indicie) % maxnphi  # Wrap around for cyclic nphi #we will assume 180 for now
                        # cyclic_unique_layer_index = unique_layer_index + dradius
                        print(f"cyclic_unique_layer_index: {cyclic_unique_layer_index}, cyclic_nphi: {cyclic_nphi}, noncyclic_z: {noncyclic_z}") if verbose else None
                        
                        # neighbor_pos = (cyclic_unique_layer_index, cyclic_nphi)
                        neighbor_pos_full = (cyclic_unique_layer_index, cyclic_nphi, noncyclic_z)


                        if len(dicNeighbors[current_pos_full]) < neighborAtLeast and neighbor_pos_full in dic_posToKey_by_batch and neighbor_pos_full not in dicNeighbors[current_pos_full]: 
                            #not already over nieghbor atleast
                            #nieghbor exists (i.e. has been fired) (then assume also exists in edep)
                            #and it hasnt already been counted in dicNeighbors
                            # numNeighbors += 1 #add for current cell to neighbor
                            print(f"neighbor for above cyclics") if verbose else None
                            
                            dicNeighbors[current_pos_full].append(neighbor_pos_full) #add neighbor to cell
                            if neighbor_pos_full not in dicNeighbors:
                                dicNeighbors[neighbor_pos_full] = []
                            dicNeighbors[neighbor_pos_full].append(current_pos_full) #add cell to neighbor (reduce double counting)
                            # numNeighbors += 1 #add for neighbor to current cell
                            
                        if len(dicEdepNeighbors[current_pos_full]) < edepNeighborAtLeast and neighbor_pos_full in dic_posToKey_by_batch and neighbor_pos_full not in dicEdepNeighbors[current_pos_full]:
                            neighborEdep = dic_posToKey_by_batch[neighbor_pos_full]['energy_dep_per_cell_RPhiZ'][0] if len(dic_posToKey_by_batch[neighbor_pos_full]['energy_dep_per_cell_RPhiZ']) == 1 else RuntimeError("check since more than one edep") #energy deposition in the cell

                            if abs(currentEdep - neighborEdep) <= edepRange: #if neighbor within range of edep
                                # numEdepNeighbors += 1 #add for current cell to neighbor
                                dicEdepNeighbors[current_pos_full].append(neighbor_pos_full) #add neighbor to cell
                                if neighbor_pos_full not in dicEdepNeighbors:
                                    dicEdepNeighbors[neighbor_pos_full] = []
                                dicEdepNeighbors[neighbor_pos_full].append(current_pos_full) #add cell to neighbor (reduce double counting)
                                # numEdepNeighbors += 1 #add for neighbor to current cell
                                
                        if len(dicNeighbors[current_pos_full]) >= neighborAtLeast:
                            neighbors = True
                        if len(dicEdepNeighbors[current_pos_full]) >= edepNeighborAtLeast:
                            neighborsEdep = True
                        # if neighbors and neighborsEdep:
                        #     print(f"already found enough neighbors for {current_pos_full}, breaking out of dphi loop") if verbose else None
                        #     break
                    
                    if len(dicGlobalNeighbors[current_global_pos_full]) < neighborAtLeast:
                        ###global search for neighbors
                        # Define box bounds
                        # lower_global_bound = [current_global_pos_full[0]-rangeR, current_global_pos_full[1]-rangePhi, current_global_pos_full[2]-rangeZ]
                        # upper_global_bounds = [current_global_pos_full[0]+rangeR, current_global_pos_full[1]+rangePhi, current_global_pos_full[2]+rangeR]
                        globalRangeR = rangeR * 14 #rough average distance betweens layers
                        globalRangePhi = rangePhi * 0.015017671 #rough average distance between phi cells
                        globalRangeZ = rangeZ #* 200
                        print(f"globalRangeR: {globalRangeR}, globalRangePhi: {globalRangePhi}, globalRangeZ: {globalRangeZ}") if i == 0 and verbose else None #if verbose else None
                        # Find all points in box
                        candidates = global_tree.query_ball_point(current_global_pos_full, r=max(globalRangeR,globalRangePhi,globalRangeZ), p=1)#p=np.inf) #May need to take into account z is both ways i.e. -2000 to 2000
                        #global R can be between ~300 to ~2500, phi can be between 0 to 2pi or ~6.28, z can be between -2000 to 2000
                        # Filter to exact box
                        result = {}
                        if len(candidates) == 0:
                            print(f"global search for neighbors, found no candidates for global: {current_global_pos_full}") if verbose else None
                            continue
                        for idx in candidates:
                            if len(dicGlobalNeighbors[current_global_pos_full]) >= neighborAtLeast:
                                break
                            print(f"candidates: {global_rphiz_points[idx]} for {current_global_pos_full}") if i < 28 and verbose else None
                            # input("Press Enter to continue... global search for neighbors \n") #if i < 28 else None
                            key = tuple(global_rphiz_points[idx])
                            # if verbose or True:
                                # print(f"checking bounds where {key[0]} - {current_global_pos_full[0]} <= {globalRangeR} and {key[1]} - {current_global_pos_full[1]} <= {globalRangePhi} and {key[2]} - {current_global_pos_full[2]} <= {abs(globalRangeZ)} which returns {abs(key[0] - current_global_pos_full[0]) <= globalRangeR, abs(key[1] - current_global_pos_full[1]) <= globalRangePhi, abs(key[2] - current_global_pos_full[2]) <= abs(globalRangeZ)}") #if (abs(key[0] - current_global_pos_full[0]) <= globalRangeR and abs(key[1] - current_global_pos_full[1]) <= globalRangePhi and abs(key[2] - current_global_pos_full[2]) <= abs(globalRangeZ)) else print(f"checking bounds where {key[0]} - {current_global_pos_full[0]} <= {globalRangeR} and {key[1]} - {current_global_pos_full[1]} <= {globalRangePhi} and {key[2]} - {current_global_pos_full[2]} <= {abs(globalRangeZ)} which returns False") if verbose else None
                            if key == current_global_pos_full: #if the key is the same as the current global pos, skip it
                                print(f"global search for neighbors, found {key}, but is the same as current global pos: {current_global_pos_full}") if verbose else None
                                continue
                            if (abs(key[0] - current_global_pos_full[0]) <= globalRangeR and 
                                abs(key[1] - current_global_pos_full[1]) <= globalRangePhi and 
                                abs(key[2] - current_global_pos_full[2]) <= abs(globalRangeZ)):
                                print(f"global search for neighbors, found {key}, with length: {len(result)} candidates for global: {current_global_pos_full}") if verbose else None
                                # result[key] = dic_posToKey_by_batch['global_rphiz_pos'][key]
                                dicGlobalNeighbors[current_global_pos_full].append(key) #add neighbor to cell
                                if key not in dicGlobalNeighbors:
                                    dicGlobalNeighbors[key] = []
                                dicGlobalNeighbors[key].append(current_global_pos_full) #add cell to neighbor (reduce double counting)
                            else:
                                print(f"global search for neighbors, found {key}, but not in range: {current_global_pos_full}") if verbose else None
                        # print(f"global search for neighbors, found {len(dicGlobalNeighbors[current_global_pos_full])} candidates for global: {current_global_pos_full} where: {dicGlobalNeighbors[current_global_pos_full]}") #if verbose else None
                        # input("Press Enter to continue...")
                    # print(f"global search for neighbors, found {dicGlobalNeighbors[current_global_pos_full]}, with length: {len(dicGlobalNeighbors[current_global_pos_full])} candidates for global: {current_global_pos_full}") #if verbose else None
                    # input("Press Enter to continue... global search for neighbors \n")
                    else:
                        print(f"already found {len(dicGlobalNeighbors[current_global_pos_full])} neighbors for global: {current_global_pos_full}") if verbose else None
                #end of phi loop
            #end of dz loop
        #end of radius loop
        print(i) if verbose else None
        
                
        print(f"finished checking neighbors for parent: {current_pos_full}, neighbors: {len(dicNeighbors[current_pos_full])}, edep neighbors: {len(dicEdepNeighbors[current_pos_full])}") if verbose else None
        #determine outcome of cell:
        if len(dicGlobalNeighbors[current_global_pos_full]) >= neighborAtLeast:
            batchVars['pos_global_only_neighbors'].append(current_global_pos_full) #add to global only neighbors
        else:
            print(f"WARNING... not enough neighbors for global pos: {current_global_pos_full}, neighbors: {len(dicGlobalNeighbors[current_global_pos_full])}, neighborAtLeast: {neighborAtLeast}") if verbose else None
            # input("Press Enter to continue... global neighbors \n") #if verbose else None
        if neighbors: #if neighbors, add to only_neighbors
            if len(dicGlobalNeighbors[current_global_pos_full]) < neighborAtLeast:
                print(f"WARNING... enough neighbors for pos: {current_pos_full}, not enough neighbors for global pos: {current_global_pos_full}, neighbors: {len(dicGlobalNeighbors[current_global_pos_full])}, neighborAtLeast: {neighborAtLeast}") #if verbose else None
                input("Press Enter to continue... neighbors \n") #if verbose else None
            batchVars['pos_only_neighbors'].append(current_pos_full)
            NeighborsRemained += 1
        else:
            NoNeighborsRemoved += 1
        if neighborsEdep:
            batchVars['pos_only_neighbors_only_edeps'].append(current_pos_full)
            EdepNeighborsRemained += 1
        else:
            NoEdepNeighborsRemoved += 1
    #end of pos hits
            
    print(f"no neighbors removed: {NoNeighborsRemoved}, no edep neighbors removed: {NoEdepNeighborsRemoved}, neighbors remained: {NeighborsRemained}, edep neighbors remained: {EdepNeighborsRemained}") #if NoNeighborsRemoved > 0 else None
    
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
    dic_all_occupancies['occupancies_per_batch_sum_batches'][dic_int_vars['numBatches']] = batch_occupancy
    
    #now determine non-neighbor occupancy
    batch_occupancy_only_neighbor = []
    batch_occupancy_xyz_only_neighbor = []
    batch_occupancy_only_neighbor_only_edep = []
    dic_int_vars, dic_posToKey_by_batch, dic_RPhiKey_by_batch, dic_list_mcIDs_one_batch, dic_all_occupancies, batchVars = calculateOnlyNeighbors(dic_int_vars, dic_posToKey_by_batch, dic_RPhiKey_by_batch, dic_list_mcIDs_one_batch, dic_all_occupancies, batchVars) 
    #for each batch, occupancies_a_batch_only_neighbor is a list of tuples but we will return a list of unique_layer_index
    no_cuts_layers = [(pos[0],) for pos in batchVars['pos']] 
        
    for unique_layer_index in range(0, dic_int_vars['total_number_of_layers']):
        batch_occupancy_only_neighbor.append(calculateOccupancy(batchVars['pos_only_neighbors'], unique_layer_index, dic_int_vars['n_cell_per_layer']))
        batch_occupancy_only_neighbor_only_edep.append(calculateOccupancy(batchVars['pos_only_neighbors_only_edeps'], unique_layer_index, dic_int_vars['n_cell_per_layer']))
        
        global_neighbor_layers = []
        global_neighbor_layers = [(pos[3],) for pos in batchVars['pos_global_only_neighbors']] #filter global neighbors by layer
        # print(f"global_neighbor_layers: {global_neighbor_layers}") if len(global_neighbor_layers) > 0 else None
        # input("Press Enter to continue... global_neighbor_layers \n") if len(global_neighbor_layers) > 0 else None
        batch_occupancy_xyz_only_neighbor.append(calculateOccupancy(global_neighbor_layers, unique_layer_index, dic_int_vars['n_cell_per_layer']))
    dic_all_occupancies['occupancies_per_batch_only_neighbors'][dic_int_vars['numBatches']] = batch_occupancy_only_neighbor
    dic_all_occupancies['occupancies_per_batch_only_neighbors_only_edeps'][dic_int_vars['numBatches']] = batch_occupancy_only_neighbor_only_edep
    dic_all_occupancies['occupancies_xyz_per_batch_only_neighbors'][dic_int_vars['numBatches']] = batch_occupancy_xyz_only_neighbor
    
    
    print(f"comparing length of pos: {len(batchVars['pos'])}, , pos_global_only_neighbors: {len(batchVars['pos_global_only_neighbors'])}")
    # if len(batchVars['pos']) != len(batchVars['pos_global_only_neighbors']):
    #     # print(f"pos: {batchVars['pos']}, pos_global_only_neighbors: {batchVars['pos_global_only_neighbors']}")
    #     for pos in no_cuts_layers:
    #         if pos not in global_neighbor_layers:
    #             print(f"pos in pos but not in pos_global_only_neighbors: {pos}")
    #     for pos in global_neighbor_layers:
    #         if pos not in no_cuts_layers:
    #             print(f"pos in pos_global_only_neighbors but not in pos: {pos}")
        
    # input("Press Enter to continue...") if len(batchVars['pos']) != len(batchVars['pos_global_only_neighbors']) else None
        
        
        
    
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
    z_layer_to_shift, drift_chamber_info, radius_to_layer = layerDrift(DCHi, n_cell_per_layer)
    z_layer_shifted_phi = globalIndicieShift(z_layer_to_shift)
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
    occupancies_keys = ["occupancies_per_batch_sum_batches", "occupancies_per_batch_only_neighbors", "occupancies_per_batch_only_neighbors_only_edeps", 
                        "avg_energy_dep_per_batch",
                        "occupancies_xyz_per_batch_sum_batches", "occupancies_xyz_per_batch_only_neighbors", 
                        "occupancies_per_batch_only_bkg", "occupancies_per_batch_only_signal", 
                        "occupancies_per_batch_only_bkg_only_neighbors", "occupancies_per_batch_only_signal_only_neighbors",
                        "occupancies_per_batch_only_bkg_only_neighbors_only_edeps", "occupancies_per_batch_only_signal_only_neighbors_only_edeps",]
    for key in occupancies_keys:
        dic_all_occupancies[key] = np.zeros((int(eventFactor*numfiles/batches), total_number_of_layers)) #we want it to be (500/20, 14) so 14 across 25 down
    
    print(f"Number of batches: {dic_all_occupancies['occupancies_per_batch_sum_batches'].shape[0]} \n") #number of batches
    
    dic_occupancies_per_batch_sum_batch_energy_dep_per_cell = {} #this will be a dictionary of np arrays
    
    dic_posToKey_by_batch_keys = ["mcID_index",
                                     "cell_fired_pos", 
                                     "shifted_phi",
                                     "energy_dep_per_cell_RPhiZ",
                                     "energy_dep_per_cell_RPhiZ_noacc",
                                     "energy_dep_per_cell_xyz_noacc",
                                     "global_xyz_pos",
                                     "global_rphiz_pos",
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
    dic_int_vars_keys = ["numBatches", "NoNeighborsRemoved", "NeighborsRemained", "EdepNeighborsRemained", "NoEdepNeighborsRemoved", "list_max_n_cell_per_layer", "list_max_n_cell_per_superlayer", "n_cell_per_layer", "n_cell_per_superlayer", "z_layer_to_shift", "drift_chamber_info", "radius_to_layer", "total_number_of_cells", "total_number_of_layers", "radiusR", "radiusPhi", "atLeast", "edepRange", "edepAtLeast", "edepLoosen", "zrange"]
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
    dic_int_vars["drift_chamber_info"] = drift_chamber_info
    dic_int_vars["radius_to_layer"] = radius_to_layer
    dic_int_vars["total_number_of_cells"] = total_number_of_cells
    dic_int_vars["total_number_of_layers"] = total_number_of_layers
    dic_int_vars["radiusR"] = radiusR
    dic_int_vars["radiusPhi"] = radiusPhi
    dic_int_vars["atLeast"] = atLeast
    dic_int_vars["edepRange"] = edepRange
    dic_int_vars["edepAtLeast"] = edepAtLeast
    dic_int_vars["edepLoosen"] = edepLoosen
    dic_int_vars["zStep"] = 100
    dic_int_vars["zpos"] = zsteps(DCHi, dic_int_vars["zStep"])
    dic_int_vars["zrange"] = zrange * dic_int_vars["zStep"]
    print(f"zrange: {zrange, dic_int_vars['zrange']}")
    
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
        # input("Press Enter to continue...") if i < 10 else None #if i < 10 else None
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
                current_cell_fired_position_tuple_xyz_to_rphiz = (np.sqrt(dc_hit.getPosition().x**2 + dc_hit.getPosition().y**2), np.mod(np.arctan2(dc_hit.getPosition().y, dc_hit.getPosition().x), 2 * np.pi), dc_hit.getPosition().z, unique_layer_index)
                # if int(current_cell_fired_position_tuple_xyz_to_rphiz[0]) not in dic_int_vars["radius_to_layer"]:
                #     print(f"Error: radius {current_cell_fired_position_tuple_xyz_to_rphiz[0]} not found in radius_to_layer dictionary")
                #     input("Press Enter to continue...")
                
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
                    
                    # batchVars['pos'].append((unique_layer_index, nphi))
                    batchVars['pos'].append(current_cell_fired_position_tuple)#esseentially replicates key of dic_posToKey_by_batch but for 
                    
                    
                    #point rphiz to global xyz
                    dic_posToKey_by_batch[current_cell_fired_position_tuple]['global_xyz_pos'].append(current_cell_fired_position_tuple_xyz)
                    
                    
                    #check if the current cell fired position tuple xyz to rphiz is unique
                    # allXYZtoRPHIZ = {}
                    # for hit in dic_posToKey_by_batch.keys():
                    #     print(f"hit: {hit}, current_cell_fired_position_tuple: {current_cell_fired_position_tuple}. current_cell_fired_position_tuple_xyz_to_rphiz: {current_cell_fired_position_tuple_xyz_to_rphiz}")
                    #     print(f"dic_posToKey_by_batch[hit]['global_rphiz_pos']: {dic_posToKey_by_batch[hit]['global_rphiz_pos']}")
                    #     if hit == current_cell_fired_position_tuple:
                    #         continue
                    #     allXYZtoRPHIZ[dic_posToKey_by_batch[hit]['global_rphiz_pos'][0]] = hit
                    # if current_cell_fired_position_tuple_xyz_to_rphiz in allXYZtoRPHIZ:
                    #     print(f"cellID_unique_identifier: {cellID_unique_identifier}, with repeat xys to rphi where: current_cell_fired_position_tuple: {current_cell_fired_position_tuple}, current_cell_fired_position_tuple_xyz: {current_cell_fired_position_tuple_xyz}, current_cell_fired_position_tuple_xyz_to_rphiz: {current_cell_fired_position_tuple_xyz_to_rphiz}")
                    #     input("Press Enter to continue...") #this is just to check if the xyz to rphiz conversion is correct, it should be unique for each cell
                    dic_posToKey_by_batch[current_cell_fired_position_tuple]['global_rphiz_pos'].append(current_cell_fired_position_tuple_xyz_to_rphiz)
                    
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
    dic["occupancy_per_batch_sum_batches"] = np.mean(dic_all_occupancies['occupancies_per_batch_sum_batches'], axis=0)
    # print(f"dic['occupancy_per_batch_sum_batches']: {dic['occupancy_per_batch_sum_batches']}")
    dic["occupancy_per_batch_sum_batches_error"] = np.std(dic_all_occupancies['occupancies_per_batch_sum_batches'], axis=0, ddof=ddofFactor) / np.sqrt(dic_all_occupancies['occupancies_per_batch_sum_batches'].shape[0])
    #error is the std of the mean, i.e. std / sqrt(n)
    dic["occupancy_per_batch_sum_batches_non_meaned"] = dic_all_occupancies['occupancies_per_batch_sum_batches']

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
    
    dic["occupancies_xyz_per_batch_only_neighbors"] = np.mean(dic_all_occupancies['occupancies_xyz_per_batch_only_neighbors'], axis=0)
    dic["occupancies_xyz_per_batch_only_neighbors_error"] = np.std(dic_all_occupancies['occupancies_xyz_per_batch_only_neighbors'], axis=0, ddof=ddofFactor) / np.sqrt(dic_all_occupancies['occupancies_xyz_per_batch_only_neighbors'].shape[0])

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
    
    
    r_shifted_phi_z = []
    for batch in list_posToKey_by_batch:
        batch_r_shifted_phi = {}
        # print(f"batch.keys(): {batch.keys()}")
        for pos_tuple in batch.keys():
            if len(batch[pos_tuple]['shifted_phi']) > 1:
                RuntimeError("shifted_phi is not a single value")
            else:
                layer = pos_tuple[0]
                nphi = pos_tuple[1]
                z = pos_tuple[2]
                # print(f"z_layer_shifted_phi: {z_layer_shifted_phi, (layer, z)}")
                shifted_nphi = z_layer_shifted_phi[(layer, z)]
                batch_r_shifted_phi[(layer,nphi + shifted_nphi,z)] = batch[pos_tuple]['energy_dep_per_cell_RPhiZ']
        r_shifted_phi_z.append(batch_r_shifted_phi)
    dic["energy_dep_per_cell_r_shifted_phi_z"] = r_shifted_phi_z #list of dictionaries; keys = pos, value = edep
    
    
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
            boolArg = True if args.calc[7] == "True" else False
            updateOcc(tpyeFile = args.calc[0], numfiles = int(args.calc[1]), radiusR = int(args.calc[2]), radiusPhi = int(args.calc[3]), atLeast = int(args.calc[4]), edepRange = float(args.calc[5]), edepAtLeast = int(args.calc[6]), edepLoosen = boolArg)
        elif args.calc[0] in typeFile and len(args.calc) == 9:
            boolArg = True if args.calc[7] == "True" else False
            updateOcc(typeFile = args.calc[0], numfiles = int(args.calc[1]), radiusR = int(args.calc[2]), radiusPhi = int(args.calc[3]), atLeast = int(args.calc[4]), edepRange = float(args.calc[5]), edepAtLeast = int(args.calc[6]), edepLoosen = boolArg, zrange = int(args.calc[8]))
        else:
            parser.error("Invalid fileType")
        # except ValueError as e:
        #     parser.error(str(e))
            
    endtime = time.time()
    print("Time taken: ", endtime - starttime)
    #'''