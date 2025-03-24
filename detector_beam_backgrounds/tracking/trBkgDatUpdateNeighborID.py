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

"""
This script is used to ...
"""

def updateOcc(typeFile="bkg", numfiles=500, radiusR=1, radiusPhi=-1, atLeast=1, edepRange=1, edepAtLeast=1, edepLoosen=-1, flexible=True):
    print("Calculating occupancy data from files...")
    list_overlay = []
    # numfiles = 500
    # typeFile = "bkg"
    # flexible = True ##basically allows the list_overlay to skip over files that dont work
    if radiusPhi == -1:
        radiusPhi = radiusR

    #setup dictionary
    dic = {}
    #can change dic_file_path to the correct path:
    if edepLoosen == -1:
        dic_file_path = "/eos/user/a/alpoulin/fccBBTrackData/wEdep/" + str(typeFile) + "_background_particles_" + str(numfiles)  + "_v6" + \
            "_R" + str(radiusR) + "_P" + str(radiusPhi) + "_AL" + str(atLeast) + "_ER" + str(edepRange) + "_EAL" + str(edepAtLeast) + ".npy" #cernbox (to save storage)
        output_dic_file_path = "/eos/user/a/alpoulin/fccBBTrackData/wEdep/" + str(typeFile) + "_background_particles_" + str(numfiles)  + "_v6" + \
            "_R" + str(radiusR) + "_P" + str(radiusPhi) + "_AL" + str(atLeast) + "_ER" + str(edepRange) + "_EAL" + str(edepAtLeast) + ".npy" #cernbox (to save storage)
    else:
        dic_file_path = "/eos/user/a/alpoulin/fccBBTrackData/wEdepL/" + str(typeFile) + "_background_particles_" + str(numfiles)  + "_v6" + \
            "_R" + str(radiusR) + "_P" + str(radiusPhi) + "_AL" + str(atLeast) + "_ER" + str(edepRange) + "_EAL" + str(edepAtLeast) + "_EL" + str(edepLoosen) + ".npy"
        output_dic_file_path = "/eos/user/a/alpoulin/fccBBTrackData/wEdepL/" + str(typeFile) + "_background_particles_" + str(numfiles)  + "_v6" + \
            "_R" + str(radiusR) + "_P" + str(radiusPhi) + "_AL" + str(atLeast) + "_ER" + str(edepRange) + "_EAL" + str(edepAtLeast) + "_EL" + str(edepLoosen) + ".npy"
    neighborID_keys = ["byBatchNeighbors", "oneDbyBatchNeighbors", ]
                    #    "oneDneighborPtN1", "oneDneighborPDGN1", "oneDneighborPtN2", "oneDneighborPDGN2", 
                    #    "oneDneighborPtN3", "oneDneighborPDGN3", "oneDneighborPtN4", "oneDneighborPDGN4", "oneDneighborPtN5", "oneDneighborPDGN5"]
    #check if dic_file_path exists:
    try:
        dic = np.load(dic_file_path, allow_pickle=True).item()
        print(f"Dictionary loaded from {dic_file_path}")
        for key in neighborID_keys:
            dic[key] = []
    except:
        print(f"Dictionary not found at {dic_file_path}")
        input("Press Enter to continue...")
        print("Creating new dictionary")
        #assign dic to empty dictionary
        dic = {}
        for key in neighborID_keys:
            dic[key] = []
        np.save(dic_file_path, dic)
        

    #we basically want to get a r,phi map where each element is the number of neighbhors (averaged across batches)
    pos_by_batch = dic["cell_fired_pos_by_batch"] #a list of tuples of (r, phi) for each cell_fired/hit
    edep_by_batch = dic["energy_dep_per_cell_per_batch"] #a list of edep for each cell_fired/hit
    pT_by_batch = dic["neighborPt_by_batch"] #a list of mcParticle indexes for each cell_fired/hit
    pdg_by_batch = dic["neighborPDG_by_batch"] #a list of pdg for each cell_fired/hit
    dic["byBatchNeighbors"] = np.zeros((len(pos_by_batch), dic["total_number_of_layers"], dic["max_n_cell_per_layer"])) 
    # dic["byBatchNeighborsEdep"] = np.zeros((len(pos_by_batch), dic["total_number_of_layers"], dic["max_n_cell_per_layer"]))
    dic["oneDbyBatchNeighbors"] = []
    dic["oneDbyBatchNeighborsEdep"] = []
    # dic["oneDneighborPtN1"] = []
    # dic["oneDneighborPDGN1"] = []
    
    dic["removedNeighborsPt"] = []
    dic["removedNeighborsPDG"] = []
    dic["removedNeighborsEdepPt"] = []
    dic["removedNeighborsEdepPDG"] = []
    
    neighbor_batch_list_of_dicts = [] #list of batches; in each batch, dictionary of neighbors
    neighborEdep_batch_list_of_dicts = []
    
    totalNeighborsRemoved = 0
    totalNeighborsRemained = 0
    totalNeighborsEdepRemoved = 0
    totalNeighborsEdepRemained = 0
    
    
    print(f"radiusR: {radiusR}, radiusPhi: {radiusPhi}, atLeast: {atLeast}, edepRange: {edepRange}, edepAtLeast: {edepAtLeast}, edepLoosen: {edepLoosen}")
    
    #initialize the array of depth(numbatches) by width (nphi) by height (layers)
    maxnphi = dic["max_n_cell_per_layer"]
    for i, batch in enumerate(pos_by_batch): #i is batch number; iterate through all batches
        print(f"batchNum: {i}")
        print(f"batch: {len(batch)}")
        
        neighbor_dic = {} #dictionary to store neighbors
        neighborEdep_dic = {} #dictionary to store neighborEdep
        
        #make the batch a set/dict to alow faster lookup
        set_batch = {}
        dic_edep_batch = {}
        for j, hit in enumerate(batch):
            set_batch[hit] = True
            # print(f"hit: {hit}")
            # print(f"eden_by_batch[i]: {edep_by_batch[i][j]}")
            if hit != (edep_by_batch[i][j][0], edep_by_batch[i][j][1]):
                #throw error
                raise ValueError("Error: hit and edep do not match")
            if hit in dic_edep_batch:
                #throw error
                raise ValueError("Error: hit already in dic_edep_batch")
            dic_edep_batch[hit] = edep_by_batch[i][j][2]
        # print(set_batch)
        # print(f"dic_edep_batch: {dic_edep_batch}")
        # input("press Enter to continue...")
        
        
        for j, hit in enumerate(batch): #j is hit number; iterate through all hits in a batch
            r, phi = hit
            edep = dic_edep_batch[hit]
            # print(edep)
            
            #if hit is not in neighbor_dic, add it
            if (r, phi) not in neighbor_dic:
                neighbor_dic[(r, phi)] = []
            if (r, phi) not in neighborEdep_dic:
                neighborEdep_dic[(r, phi)] = []
            
            #get the number of neighbors
            numNeighbors = 0
            numEdepNeighbors = 0
            for dx in range(-radiusR, radiusR+1):
                for dy in range(-radiusPhi, radiusPhi+1):
                    if dx == 0 and dy == 0:
                        continue
                    cyclic_phi = (phi + dy) % maxnphi
                    if (r+dx, cyclic_phi) in neighbor_dic and (r+dx, cyclic_phi) in neighborEdep_dic[(r, phi)]:
                        continue #in both skip
                    
                    if (r+dx, cyclic_phi) in set_batch and (r+dx, cyclic_phi) not in neighbor_dic[(r, phi)]: #fired and not already accounted for
                        if (r+dx, cyclic_phi) not in neighbor_dic:
                            neighbor_dic[(r+dx, cyclic_phi)] = []
                        neighbor_dic[(r, phi)].append((r+dx, cyclic_phi)) #add the neighbor to the current hit
                        neighbor_dic[(r+dx, cyclic_phi)].append((r, phi)) #add the current hit to the neighbor
                        
                    if (r+dx, cyclic_phi) in dic_edep_batch and (r+dx, cyclic_phi) not in neighborEdep_dic[(r,phi)] and abs(dic_edep_batch[(r+dx, cyclic_phi)] - dic_edep_batch[(r, phi)]) <= edepRange:
                        if (r+dx, cyclic_phi) not in neighborEdep_dic:
                            neighborEdep_dic[(r+dx, cyclic_phi)] = []
                        neighborEdep_dic[(r, phi)].append((r+dx, cyclic_phi, dic_edep_batch[(r+dx, cyclic_phi)])) #add the neighbor to the current hit
                        neighborEdep_dic[(r+dx, cyclic_phi)].append((r, phi, edep)) #add the current hit to the neighbor
                #end dy loop
            #end dx loop
            #went through all possible neighbors for current hit
            # print(f"num neighbors for hit {hit}: {len(neighbor_dic[(r, phi)])}")
            numNeighbors = len(neighbor_dic[(r, phi)])
            numEdepNeighbors = len(neighborEdep_dic[(r, phi)])
            
            
            if numNeighbors == 0:
                dic["removedNeighborsPt"].append(pT_by_batch[i][j])
                dic["removedNeighborsPDG"].append(pdg_by_batch[i][j])
                totalNeighborsRemoved += 1
            else:
                totalNeighborsRemained += 1
            if numEdepNeighbors == 0:
                dic["removedNeighborsEdepPt"].append(pT_by_batch[i][j])
                dic["removedNeighborsEdepPDG"].append(pdg_by_batch[i][j])
                totalNeighborsEdepRemoved += 1
            else:
                totalNeighborsEdepRemained += 1
                        
                        
            # print(hist["byBatchNeighbors"][i].shape)
            # input("press Enter to continue...")
            dic["byBatchNeighbors"][i][r, phi] = numNeighbors
            dic["oneDbyBatchNeighbors"].append(numNeighbors) #for every hit we want to get the neighbors of that cell fired
            dic["oneDbyBatchNeighborsEdep"].append(numEdepNeighbors) #for every hit we want to get the neighbors of that cell fired
            # if numNeighbors == 1:
            #     dic["oneDneighborPtN1"].append(pT_by_batch[i][j])
            #     dic["oneDneighborPDGN1"].append(pdg_by_batch[i][j])
            # if numNeighbors == 2:
            #     dic["oneDneighborPtN2"].append(pT_by_batch[i][j])
            #     dic["oneDneighborPDGN2"].append(pdg_by_batch[i][j])
            # if numNeighbors == 3:
            #     dic["oneDneighborPtN3"].append(pT_by_batch[i][j])
            #     dic["oneDneighborPDGN3"].append(pdg_by_batch[i][j])
            # if numNeighbors == 4:
            #     dic["oneDneighborPtN4"].append(pT_by_batch[i][j])
            #     dic["oneDneighborPDGN4"].append(pdg_by_batch[i][j])
            # if numNeighbors == 5:
            #     dic["oneDneighborPtN5"].append(pT_by_batch[i][j])
            #     dic["oneDneighborPDGN5"].append(pdg_by_batch[i][j])
                
        #end of all hits in a batch
        
        # neighbor_batch_list_of_dicts.append(neighbor_dic)
        # neighborEdep_batch_list_of_dicts.append(neighborEdep_dic)
    #end of all batches
    
    
    # print(hist["byBatchNeighbors"].shape)
    #average across the depth (aka the batches)
    # dic["byBatchNeighborsAvg"] = np.mean(dic["byBatchNeighbors"], axis=0)
    # dic["byBatchNeighborsMedian"] = np.median(dic["byBatchNeighbors"], axis=0)
    
    # dic["neighbor_batch_list_of_dicts"] = neighbor_batch_list_of_dicts
    # dic["neighborEdep_batch_list_of_dicts"] = neighborEdep_batch_list_of_dicts


    print("finished updating dictionary")
    
    print(f"Total neighbors removed: {totalNeighborsRemoved}")
    print(f"Total neighbors remained: {totalNeighborsRemained}")
    print(f"Total neighbors edep removed: {totalNeighborsEdepRemoved}")
    print(f"Total neighbors edep remained: {totalNeighborsEdepRemained}")


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
                        "\n-- edepAtLeast(int): Default(1)",
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
                boolArg = 1 if args.calc[7] == "True" else 0
                updateOcc(args.calc[0], int(args.calc[1]), int(args.calc[2]), int(args.calc[3]), int(args.calc[4]), float(args.calc[5]), int(args.calc[6]), boolArg)
            else:
                parser.error("Invalid fileType")
        except ValueError as e:
            parser.error(str(e))
            
    endtime = time.time()
    print(f"Time taken: {endtime - starttime} seconds")
    #'''