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
import ROOT
import sys, os

list_overlay = []
number_of_bx = 500
number_of_iteration_on_bx_batches = 1
bx_batch_index = 0
bunch_spacing = 20 # ns

# oldDataPath = "/ceph/submit/data/group/fcc/ee/detector/tracking/IDEA_background_only/"
bkgDataPath = "/ceph/submit/data/group/fcc/ee/detector/tracking/IDEA_background_only_IDEA_o1_v03_v3/" #kinetic threshold
# combinedDataPath = "/ceph/submit/data/group/fcc/ee/detector/tracking/Zcard_CLD_background_v1/" #needs newer key4hep
combinedDataPath = "/ceph/submit/data/group/fcc/ee/detector/tracking/Zcard_CLD_background_IDEA_o1_v03_v4" #newer beampipe
# combinedDataPath = "/ceph/submit/data/group/fcc/ee/detector/tracking/Zcard_CLD_background_IDEA_o1_v03_v4" #just signal #scratch/Zcard_CLD_background_IDEA_o1_v03_v4
bkgFilePath = "out_sim_edm4hep_background_"
combinedFilePath = "/out_sim_edm4hep"
signalFilePath = "/out_sim_edm4hep_base"
type = "bkg"

for i in range(1,number_of_bx + 1):
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
dic_file_path = "fccproject-tracking/detector_beam_backgrounds/tracking/data/" + str(type) + "_background_particles_" + str(number_of_bx) + ".npy"
plot_folder = "fccproject-tracking/detector_beam_backgrounds/tracking/images/occupancy"

def draw_histo(histo, drawOptions = "", path = plot_folder, logY = False):
    name = histo.GetName()
    canvas = ROOT.TCanvas(name, name)
    if "colz" in drawOptions:
        histo.SetStats(0)
        canvas.SetRightMargin(0.15)
        if "map" in name:
            canvas.SetGrid()
        if "logz" in drawOptions:
            canvas.SetLogz()
            name += "_logz"
    histo.Draw(drawOptions)
    if logY:
        canvas.SetLogy()
        name += "_logy"
    canvas.Print(os.path.join(path, name + ".png"))

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


# how many cells are fired by a bkg particle?
n_cell_fired_of_particles_hitting_dch =  ROOT.TH1F(f"n_cell_fired_of_particles_hitting_dch", f"n_cell_fired_of_particles_hitting_dch", 50, 0, 50)
n_cell_fired_of_particles_hitting_dch.SetTitle(f"Number of cell fired by particles hitting the DCH")
n_cell_fired_of_particles_hitting_dch.GetXaxis().SetTitle("Number of cell fired by the particle")
n_cell_fired_of_particles_hitting_dch.GetYaxis().SetTitle("Number of entries")
# how many cells are fired by a bkg particle in log scale?
n_cell_fired_of_particles_hitting_dch_log =  ROOT.TH1F(f"n_cell_fired_of_particles_hitting_dch_log", f"n_cell_fired_of_particles_hitting_dch_log", 50, 0, 150)
n_cell_fired_of_particles_hitting_dch_log.SetTitle(f"Number of cell fired by particles hitting the DCH")
n_cell_fired_of_particles_hitting_dch_log.GetXaxis().SetTitle("Number of cell fired by the particle")
n_cell_fired_of_particles_hitting_dch_log.GetYaxis().SetTitle("Number of entries")
###


# variable to make averaged plots over many bx's
occupancy_per_layer_profile = ROOT.TProfile("occupancy_per_layer_profile", "occupancy_per_layer_profile", 112, 0, 112, "s")
occupancy_per_layer_profile.SetTitle(f"Average occupancy per layer, {number_of_bx} BXs ran {number_of_iteration_on_bx_batches} times")
occupancy_per_layer_profile.GetXaxis().SetTitle("Radial layer index")
occupancy_per_layer_profile.GetYaxis().SetTitle("Average channel occupancy [%]")
occupancies_per_layer = []
overall_occupancies = []

total_number_of_hit = 0
total_number_of_hit_comingFromPrimaryAfterBeamPipe = 0

string_for_overall_occupancies = ""

# cross check the number of cell
dict_layer_phiSet = {}


dict_cellID_nHits = {}
total_number_of_hit_integrated_per_batch = 0
number_of_cell_with_multiple_hits = 0

# vairable to make averaged plots over many bx's
list_occupancy_per_layer = []

# what fraction of particles do reach the drift chamber?
total_number_of_particles_reaching_dch_per_bx_batch = 0
total_number_of_particles_per_bx_batch = 0

# Histogram declarations
# what is the occupancy per radial layer?
occupancy_per_layer =  ROOT.TH1F(f"occupancy_per_layer_bx_batch_index_{bx_batch_index}", f"occupancy_per_layer_bx_batch_index_{bx_batch_index}", total_number_of_layers, 0, total_number_of_layers)
occupancy_per_layer.SetTitle(f"Occupancy per layer, {number_of_bx} BXs ({number_of_bx * bunch_spacing} ns)")
occupancy_per_layer.GetXaxis().SetTitle("Radial layer index")
occupancy_per_layer.GetYaxis().SetTitle("Channel occupancy [%]")

# Map of the fired cells energies
DC_fired_cell_map = ROOT.TH2F(f"DC_fired_cell_map_bx_batch_index_{bx_batch_index}", "DC_fired_cell_map_bx_batch_index_{bx_batch_index}", max_n_cell_per_layer, 0, max_n_cell_per_layer, total_number_of_layers, 0, total_number_of_layers) 
DC_fired_cell_map.SetTitle(f"R-phi map of fired cells (energy in MeV on z axis) ({number_of_bx} BXs)")
DC_fired_cell_map.GetXaxis().SetTitle("Cell phi index")
DC_fired_cell_map.GetYaxis().SetTitle("Cell layer index")
DC_fired_cell_map.GetZaxis().SetTitle("Energy [MeV]")


    
    
    
#loop over all the files
for i in range(0, len(list_overlay)):
    
    
    
    
    

    rootfile = list_overlay[i]
    print(f"Running over file: {rootfile}")
    reader = root_io.Reader(rootfile)
    metadata = reader.get("metadata")[0]
    if type == "combined" or type == "signal":
        cellid_encoding = metadata.get_parameter("CDCHHits__CellIDEncoding")
    else:
        cellid_encoding = metadata.get_parameter("DCHCollection__CellIDEncoding")
    decoder = dd4hep.BitFieldCoder(cellid_encoding)
    
    
    
    
    bx_seen = 0
    for event in reader.get("events"):
        bx_seen += 1 # now there is one event per rootfile representing 1 bx, may change in future so place the counter here
        if type == "combined" or type == "signal":
            dc_hits = event.get("CDCHHits")
        else:
            dc_hits = event.get("DCHCollection")
            
        # Map of the fired cells energies, per "event"
        DC_fired_cell_map_per_evt = ROOT.TH2F(f"DC_fired_cell_map_per_evt_{bx_seen}", f"DC_fired_cell_map_per_evt_{bx_seen}", max_n_cell_per_layer, 0, max_n_cell_per_layer, total_number_of_layers, 0, total_number_of_layers)
        DC_fired_cell_map_per_evt.SetTitle(f"R-phi map of fired cells (energy in MeV on z axis) ({number_of_bx} BXs)")
        DC_fired_cell_map_per_evt.GetXaxis().SetTitle("Cell phi index")
        DC_fired_cell_map_per_evt.GetYaxis().SetTitle("Cell layer index")
        DC_fired_cell_map_per_evt.GetZaxis().SetTitle("Energy [MeV]")
        total_number_of_hit_thisbx = 0
        seen_particle_ids = []
        dict_particle_n_fired_cell = {}
        dict_particle_fired_cell_id = {}
        total_number_of_particles_per_bx_batch += len(event.get("MCParticles"))
        
        for num_hit, dc_hit in enumerate(dc_hits):
            particle = dc_hit.getMCParticle()
            index_mc = particle.getObjectID().index
            
            total_number_of_hit += 1
            total_number_of_hit_integrated_per_batch += 1
            total_number_of_hit_thisbx += 1
            cellID = dc_hit.getCellID()
            layer = decoder.get(cellID, "layer")
            superlayer = decoder.get(cellID, "superlayer")
            nphi = decoder.get(cellID, "nphi")
            
            # define a unique layer index based on super layer and layer
            if layer >= n_layers_per_superlayer or superlayer >= n_superlayers:
                print("Error: layer or super layer index out of range")
                print(f"Layer: {layer} while max layer is {n_layers_per_superlayer - 1}. Superlayer: {superlayer} while max superlayer is {n_superlayers - 1}.")
            unique_layer_index = superlayer * n_layers_per_superlayer + layer
            # cross check the number of cell
            if not unique_layer_index in dict_layer_phiSet.keys():
                dict_layer_phiSet[unique_layer_index] = {nphi}
            else:
                dict_layer_phiSet[unique_layer_index].add(nphi)
            cellID_unique_identifier = "SL_" + str(superlayer)  + "_L_" + str(layer) + "_nphi_" + str(nphi) 
            # what is the occupancy?
            if not cellID_unique_identifier in dict_cellID_nHits.keys(): # the cell was not fired yet
                occupancy_per_layer.Fill(unique_layer_index)
                dict_cellID_nHits[cellID_unique_identifier] = 1
            else:
                if(dict_cellID_nHits[cellID_unique_identifier] == 1):
                    number_of_cell_with_multiple_hits += 1
                dict_cellID_nHits[cellID_unique_identifier] += 1
            # deal with the number of cell fired per particle
            particle_object_id = particle.getObjectID().index
            if particle_object_id not in dict_particle_n_fired_cell.keys(): # the particle was not seen yet
                dict_particle_n_fired_cell[particle_object_id] = 1
                dict_particle_fired_cell_id[particle_object_id] = [cellID_unique_identifier]
            else: # the particle already fired cells
                if not cellID_unique_identifier in dict_particle_fired_cell_id[particle_object_id]: # this cell was not yet fired by this particle
                    dict_particle_n_fired_cell[particle_object_id] += 1
                    dict_particle_fired_cell_id[particle_object_id].append(cellID_unique_identifier)
                    
            # Map of the fired cells energies
                DC_fired_cell_map.Fill(nphi, unique_layer_index, 1e+3*dc_hit.getEDep())
                DC_fired_cell_map_per_evt.Fill(nphi, unique_layer_index, 1e+3*dc_hit.getEDep())

    
    #print(f"dict_particle_n_fired_cell: {dict_particle_n_fired_cell}")
    for particleKey in dict_particle_n_fired_cell.keys():
        n_cell_fired_of_particles_hitting_dch.Fill(dict_particle_n_fired_cell[particleKey])
        n_cell_fired_of_particles_hitting_dch_log.Fill(dict_particle_n_fired_cell[particleKey])
                
                
                
    percentage_of_fired_cells = 100 * len(dict_cellID_nHits.keys())/float(total_number_of_cells)
  
# Normalize the occupancy per layer th1 (divide the number of cell fired by the total number of cell) and fill the TProfile of occupancies
for unique_layer_index in range(0, total_number_of_layers):
    raw_bin_content = occupancy_per_layer.GetBinContent(unique_layer_index + 1) # NB: we use the trick that the bin index here is the same as the layer index it corresponds to, just binIdx 0 is underflow
    occupancy_per_layer.SetBinContent(unique_layer_index + 1, 100 * raw_bin_content/float(n_cell_per_layer[str(unique_layer_index)])) # unique_layer_index and n_cell_per_layer key definitions coincide
    occupancy_per_layer_profile.Fill(unique_layer_index, 100 * raw_bin_content/float(n_cell_per_layer[str(unique_layer_index)]))
draw_histo(occupancy_per_layer)
overall_occupancies.append(percentage_of_fired_cells)


draw_histo(n_cell_fired_of_particles_hitting_dch)
draw_histo(n_cell_fired_of_particles_hitting_dch_log)

#draw n_cells_per_layer
    
    
# draw averaged quantities
overall_occupancies_th1 = ROOT.TH1F(f"overall_occupancies_th1", f"overall_occupancies_th1", 50, 0, 5)
overall_occupancies_th1.SetTitle(f"DC overall occupancy, {number_of_bx} BXs ran {number_of_iteration_on_bx_batches} times")
overall_occupancies_th1.GetXaxis().SetTitle("Overall occupancy [%]")
overall_occupancies_th1.GetYaxis().SetTitle("Number of entries")
for occupancy in overall_occupancies:
    overall_occupancies_th1.Fill(occupancy)
draw_histo(overall_occupancies_th1)