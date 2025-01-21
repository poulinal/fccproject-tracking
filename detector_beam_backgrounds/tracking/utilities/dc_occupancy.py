import os, sys, glob
import ROOT
from podio import root_io
import dd4hep as dd4hepModule
from ROOT import dd4hep

from math import sqrt

ROOT.gROOT.SetBatch(True)
#ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetTextSize(22)


#bkg_file_directory = "/eos/experiment/fcc/users/b/brfranco/background_files/guineaPig_andrea_Apr2024/sim_noboost_noBfield/"
#bkg_file_directory = "/eos/experiment/fcc/users/b/brfranco/background_files/guineaPig_andrea_Apr2024/sim_noboost_bfield/"
#bkg_file_directory = "/eos/experiment/fcc/users/b/brfranco/background_files/guineaPig_andrea_Apr2024/sim_boost_noBfield/"
#bkg_file_directory = "/eos/experiment/fcc/users/b/brfranco/background_files/guineaPig_andrea_Apr2024/sim_boost_bfield/"
#bkg_file_directory =  "/eos/experiment/fcc/users/b/brfranco/background_files/guineaPig_andrea_Apr2024/sim_boost_bfield_k4geo_version/"
#bkg_file_directory =  "/eos/experiment/fcc/users/b/brfranco/background_files/guineaPig_andrea_Apr2024/sim_boost_bfield_k4geo_version_Geant4TrackerAction_edep0/"
# NEW beampie 
#bkg_file_directory = "/eos/experiment/fcc/users/b/brfranco/background_files/guineaPig_andrea_June2024_v23/ddsimoutput/new_beampipe_default_ddsim_config/"
#bkg_file_directory = "/eos/experiment/fcc/users/b/brfranco/background_files/guineaPig_andrea_June2024_v23/ddsimoutput/new_beampipe_default_ddsim_config_addLumiCalAndBeamInstrumentation/"
#bkg_file_directory = "/eos/experiment/fcc/users/b/brfranco/background_files/guineaPig_andrea_June2024_v23_vtx000/ddsimoutput/guineaPig_andrea_June2024_v23_vtx000/"
# Fix in Geant4
#bkg_file_directory = "/eos/experiment/fcc/users/b/brfranco/background_files/guineaPig_andrea_June2024_v23/ddsimoutput_afterG4Fix/June2024_v23_afterG4Fix/"
#bkg_file_directory = "/eos/experiment/fcc/users/b/brfranco/background_files/guineaPig_andrea_June2024_v23/ddsimoutput_afterG4Fix_CADbeampipe/June2024_v23_afterG4Fix_CADbeampipe/"
#bkg_file_directory = "/eos/experiment/fcc/users/b/brfranco/background_files/guineaPig_andrea_June2024_v23/ddsimoutput_afterG4Fix_CADbeampipe_keepAllParticles/June2024_v23_afterG4Fix_CADbeampipe_keepAllParticles/"
bkg_file_directory = "/eos/experiment/fcc/users/b/brfranco/background_files/guineaPig_andrea_June2024_v23/ddsimoutput_afterG4Fix_CADbeampipe_keepAllParticles/June2024_v23_afterG4Fix_CADbeampipe_keepAllParticles/"
#bkg_file_directory = "/eos/experiment/fcc/users/b/brfranco/background_files/guineaPig_andrea_June2024_v23/ddsimoutput_afterG4Fix_CADbeampipe_noCopperCrotch/June2024_v23_afterG4Fix_CADbeampipe_noCopperCrotch/"
file_name_template = "IDEA_o1_v03_*.root"
#plot_folder = "plots_new_beampipe_default_ddsim_config_addLumiCalAndBeamInstrumentation_integrationTime400ns_G4Fix_logzHitOrigin"
#plot_folder = "plots_10Gev_mu_100_evt_tryFixObjID"
#plot_folder = "plots_new_beampipe_default_ddsim_config_addLumiCalAndBeamInstrumentation_integrationTime400ns_G4Fix_CADbeampipe_keepAllParticles"
#plot_folder = "plots_sr_radiation_without_mask"
#plot_folder = "plots_sr_radiation_sr_photons_from_positron_45GeVcom_halo_v23_mediumfilter_nominal"
#plot_folder = "plots_sr_radiation_sr_photons_from_positron_182GeVcom_halo_v23_mediumfilter_nominal"
#plot_folder = "plots_sr_radiation_sr_photons_from_positron_182GeVcom_halo_v23_mediumfilter_noMask_correct"
plot_folder = "plots_100MeV_electrons_IDEAo1_v03_100evt_"
# Change here for SINGLE FILE (number_of_iteration_on_bx_batches = 1 and number_of_bx = 1)
number_of_iteration_on_bx_batches = 1 # max 199
number_of_bx = 1
bunch_spacing = 20 # ns
if not os.path.isdir(plot_folder):
    os.mkdir(plot_folder)

single_bx_files = glob.glob(os.path.join(bkg_file_directory, file_name_template))
# filter possible hadded files
for single_bx_file in single_bx_files:
    file_index = os.path.basename(single_bx_file).split("_")[-1].replace(".root", "")
    if not file_index.isnumeric():
        print(f"INFO: removing {single_bx_file} from the file list because it looks like an hadded file")
        single_bx_files.remove(single_bx_file)

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

# Histogram definition
# what is the energy of the particles actually hitting the DC
energy_of_particles_hitting_dch =  ROOT.TH1F(f"energy_of_particles_hitting_dch", f"energy_of_particles_hitting_dch", 50, 0, 0.1)
energy_of_particles_hitting_dch.SetTitle(f"Energy of bkg particles hitting the DCH ({number_of_iteration_on_bx_batches * number_of_bx} BXs)")
energy_of_particles_hitting_dch.GetXaxis().SetTitle("Energy [GeV]")
energy_of_particles_hitting_dch.GetYaxis().SetTitle("Number of entries")
# what is the energy of the particles actually hitting the DC - low values
energy_of_particles_hitting_dch_lowValues =  ROOT.TH1F(f"energy_of_particles_hitting_dch_lowValues", f"energy_of_particles_hitting_dch_lowValues", 50, 0, 0.001)
energy_of_particles_hitting_dch_lowValues.SetTitle(f"Energy of bkg particles hitting the DCH ({number_of_iteration_on_bx_batches * number_of_bx} BXs)")
energy_of_particles_hitting_dch_lowValues.GetXaxis().SetTitle("Energy [GeV]")
energy_of_particles_hitting_dch_lowValues.GetYaxis().SetTitle("Number of entries")
# what is the energy of the particles actually hitting the DC when they come from photons
energy_of_particles_hitting_dch_with_photon_parent =  ROOT.TH1F(f"energy_of_particles_hitting_dch_with_photon_parent", f"energy_of_particles_hitting_dch_with_photon_parent", 50, 0, 0.1)
energy_of_particles_hitting_dch_with_photon_parent.SetTitle(f"Energy of photon originating particles hitting the DCH ({number_of_iteration_on_bx_batches * number_of_bx} BXs)")
energy_of_particles_hitting_dch_with_photon_parent.GetXaxis().SetTitle("Energy [GeV]")
energy_of_particles_hitting_dch_with_photon_parent.GetYaxis().SetTitle("Number of entries")
# what is the pt of the particles actually hitting the DC and with vertex radius below 10 cm
pt_of_particles_hitting_dch_below_10cm = ROOT.TH1F(f"pt_of_particles_hitting_dch_below_10cm", f"pt_of_particles_hitting_dch_below_10cm", 50, 0, 0.1)
pt_of_particles_hitting_dch_below_10cm.SetTitle(f"P_{{T}} of bkg particles hitting the DCH and with VTX radius < 10 cm ({number_of_iteration_on_bx_batches * number_of_bx} BXs)")
pt_of_particles_hitting_dch_below_10cm.GetXaxis().SetTitle("P_{T} [GeV]")
pt_of_particles_hitting_dch_below_10cm.GetYaxis().SetTitle("Number of entries")
# what is the de/DX of the background particles/muons (from another rootfile)?
dedx_of_dch_hits =  ROOT.TH1F(f"dedx_of_dch_hits", f"dedx_of_dch_hits", 50, 0, 1)
dedx_of_dch_hits.SetTitle(f"dE/dx of DCH hits")
dedx_of_dch_hits.GetXaxis().SetTitle("dE/dx [keV/mm]")
dedx_of_dch_hits.GetYaxis().SetTitle("Number of entries")
# what is the particle pdgid hitting DC?
pdgid_of_particles_hitting_dch =  ROOT.TH1F(f"pdgid_of_particles_hitting_dch", f"pdgid_of_particles_hitting_dch", 46, -23, 23)
pdgid_of_particles_hitting_dch.SetTitle(f"PDG ID of bkg particles hitting the DCH")
pdgid_of_particles_hitting_dch.GetXaxis().SetTitle("PDG ID")
pdgid_of_particles_hitting_dch.GetYaxis().SetTitle("Arbitrary units")
# most particles leaving signals in DCH are electrons or positrons (because photon kick out electron that then leave a signal), is their parent a photon?
hasPhotonParent_of_particles_hitting_dch =  ROOT.TH1F(f"hasPhotonParent_of_particles_hitting_dch", f"hasPhotonParent_of_particles_hitting_dch", 2, 0, 2)
hasPhotonParent_of_particles_hitting_dch.SetTitle(f"Was the particle produced by a photon?")
hasPhotonParent_of_particles_hitting_dch.GetXaxis().SetTitle("Has photon parent?")
hasPhotonParent_of_particles_hitting_dch.GetYaxis().SetTitle("Arbitrary units")
# most particles leaving signals in DCH are electrons or positrons (because photon kick out electron that then leave a signal), is their parent a photon or is it a photon itself?
hasPhotonParentOrIsPhoton_of_particles_hitting_dch =  ROOT.TH1F(f"hasPhotonParentOrIsPhoton_of_particles_hitting_dch", f"hasPhotonParentOrIsPhoton_of_particles_hitting_dch", 2, 0, 2)
hasPhotonParentOrIsPhoton_of_particles_hitting_dch.SetTitle(f"Was the particle produced by a photon?")
hasPhotonParentOrIsPhoton_of_particles_hitting_dch.GetXaxis().SetTitle("Has photon parent?")
hasPhotonParentOrIsPhoton_of_particles_hitting_dch.GetYaxis().SetTitle("Arbitrary units")
# are the particles hitting dc primaries or secondaries? Filled once per MCParticle (ill defined currently because if the secondary is not kept in the output the hit points to the primary)
isPrimary_of_particles_hitting_dch =  ROOT.TH1F(f"isPrimary_of_particles_hitting_dch", f"isPrimary_of_particles_hitting_dch", 2, 0, 2)
isPrimary_of_particles_hitting_dch.SetTitle(f"Is the particle a primary?")
isPrimary_of_particles_hitting_dch.GetXaxis().SetTitle("Is primary?")
isPrimary_of_particles_hitting_dch.GetYaxis().SetTitle("Arbitrary units")
# are the hit from the DC coming from a primary particle or not? # w.r.t. to isPrimary_of_particles_hitting_dch, this one is filled once per hit
isPrimary_of_particle_attached_to_hits =  ROOT.TH1F(f"isPrimary_of_particle_attached_to_hits", f"isPrimary_of_particle_attached_to_hits", 2, 0, 2)
isPrimary_of_particle_attached_to_hits.SetTitle(f"Is the particle a primary?")
isPrimary_of_particle_attached_to_hits.GetXaxis().SetTitle("Is primary?")
isPrimary_of_particle_attached_to_hits.GetYaxis().SetTitle("Arbitrary units")
# what is the vertex radius of the oldest parent (original primary particle)?
oldestParentVertexRadius_of_particles_hitting_dch =  ROOT.TH1F(f"oldestParentVertexRadius_of_particles_hitting_dch", f"oldestParentVertexRadius_of_particles_hitting_dch", 50, 0, 50)
oldestParentVertexRadius_of_particles_hitting_dch.SetTitle(f"Radius of oldest parent for particles hitting the DCH")
oldestParentVertexRadius_of_particles_hitting_dch.GetXaxis().SetTitle("Radius [mm]")
oldestParentVertexRadius_of_particles_hitting_dch.GetYaxis().SetTitle("Number of entries")
# where are the particles leading to hits in the DC coming from?
particle_hittingDC_origin_rz = ROOT.TH2F("particle_hittingDC_origin_rz", "particle_hittingDC_origin_rz", 100, 0, 2500, 100, 0, 200)
particle_hittingDC_origin_rz.SetTitle(f"Origin of bkg particles hitting the DCH ({number_of_iteration_on_bx_batches * number_of_bx} BXs)")
particle_hittingDC_origin_rz.GetXaxis().SetTitle("z [mm]")
particle_hittingDC_origin_rz.GetYaxis().SetTitle("r [mm]")
# where are the particles leading to hits in the DC coming from in x vs y?
particle_hittingDC_origin_xy = ROOT.TH2F("particle_hittingDC_origin_xy", "particle_hittingDC_origin_xy", 200, -2000, 2000, 200, -2000, 2000)
particle_hittingDC_origin_xy.SetTitle(f"Origin of bkg particles hitting the DCH ({number_of_iteration_on_bx_batches * number_of_bx} BXs)")
particle_hittingDC_origin_xy.GetXaxis().SetTitle("x [mm]")
particle_hittingDC_origin_xy.GetYaxis().SetTitle("y [mm]")
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
###
# MC Particle loop related definition
pt_all_electrons =  ROOT.TH1F(f"pt_all_electrons", f"pt_all_electrons", 50, 0, 1)
pt_all_electrons.SetTitle(f"P_{{T}} of all electrons")
pt_all_electrons.GetXaxis().SetTitle("GeV")
pt_all_electrons.GetYaxis().SetTitle("Number of entries")

pt_primary_electrons =  ROOT.TH1F(f"pt_primary_electrons", f"pt_primary_electrons", 50, 0, 1)
pt_primary_electrons.SetTitle(f"P_{{T}} of primary electrons")
pt_primary_electrons.GetXaxis().SetTitle("GeV")
pt_primary_electrons.GetYaxis().SetTitle("Number of entries")


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

bx_seen = 0
for bx_batch_index in range(0, number_of_iteration_on_bx_batches): # first loop to run X times on Y BX's
    print("Iteration number ", bx_batch_index)
    bx_files_seen_so_far = []
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

    # what is the energy deposited in cells, averaged over phi, per radial layer? #TODO
    cellEnergy_per_layer =  ROOT.TH1F(f"cellEnergy_per_layer_bx_batch_index_{bx_batch_index}", f"cellEnergy_per_layer_bx_batch_index_{bx_batch_index}", total_number_of_layers, 0, total_number_of_layers)
    cellEnergy_per_layer.SetTitle(f"Cell energy per layer, {number_of_bx} BXs ({number_of_bx * bunch_spacing} ns)")
    cellEnergy_per_layer.GetXaxis().SetTitle("Radial layer index")
    cellEnergy_per_layer.GetYaxis().SetTitle("Energy per cell [MeV]")

    # Where do the particles hit the DC?
    DC_simhit_position_rz = ROOT.TH2F(f"DC_simhit_position_rz_bx_batch_index_{bx_batch_index}", "DC_simhit_position_rz_bx_batch_index_{bx_batch_index}", 100, 0, 2000, 100, 0, 2000)
    DC_simhit_position_rz.SetTitle(f"DCH simHit position ({number_of_bx} BXs)")
    DC_simhit_position_rz.GetXaxis().SetTitle("z [mm]")
    DC_simhit_position_rz.GetYaxis().SetTitle("r [mm]")
    DC_simhit_position_rz.GetZaxis().SetTitle("Number of entries")

    DC_simhit_position_xy = ROOT.TH2F(f"DC_simhit_position_xy_bx_batch_index_{bx_batch_index}", "DC_simhit_position_xy_bx_batch_index_{bx_batch_index}", 200, -2000, 2000, 200, -2000, 2000)
    DC_simhit_position_xy.SetTitle(f"DCH simHit position ({number_of_bx} BXs)")
    DC_simhit_position_xy.GetXaxis().SetTitle("x [mm]")
    DC_simhit_position_xy.GetYaxis().SetTitle("y [mm]")
    DC_simhit_position_xy.GetZaxis().SetTitle("Number of entries")

    # Map of the fired cells energies
    DC_fired_cell_map = ROOT.TH2F(f"DC_fired_cell_map_bx_batch_index_{bx_batch_index}", "DC_fired_cell_map_bx_batch_index_{bx_batch_index}", max_n_cell_per_layer, 0, max_n_cell_per_layer, total_number_of_layers, 0, total_number_of_layers) 
    DC_fired_cell_map.SetTitle(f"R-phi map of fired cells (energy in MeV on z axis) ({number_of_bx} BXs)")
    DC_fired_cell_map.GetXaxis().SetTitle("Cell phi index")
    DC_fired_cell_map.GetYaxis().SetTitle("Cell layer index")
    DC_fired_cell_map.GetZaxis().SetTitle("Energy [MeV]")

    for single_bx_file in single_bx_files:
        if single_bx_file in bx_files_seen_so_far:
            print("Error: already ran over this BX! Exitting...")
            sys.exit(1)
        if len(bx_files_seen_so_far) == number_of_bx:
            print(f"Reached the number of BX needed {number_of_bx}")
            break
        bx_files_seen_so_far.append(single_bx_file)
        single_bx_files.remove(single_bx_file) # make sure we do not re-run always on the same files
        input_file_path = single_bx_file
        # change here for single file run
        #input_file_path = "/afs/cern.ch/user/b/brfranco/work/public/background_studies/k4geo/10Gev_mu_100_evt.root"
        #input_file_path = "/eos/experiment/fcc/users/b/brfranco/background_files/sr_photons_kevin/SR_photon_v3_fixed_topThreshold_tightFilter_lessMacroParticle_usedForMDITalk.root"
        #input_file_path = "/eos/experiment/fcc/users/b/brfranco/background_files/sr_photons_kevin/SR_photon_v3_fixed_topThreshold_tightFilter_lessMacroParticle_usedForMDITalk_noSRMask.root"
        #input_file_path = "/eos/experiment/fcc/users/b/brfranco/background_files/sr_photons_kevin/SR_photons_v5/sr_photons_from_positron_182GeVcom_nzco_6urad_v23_mediumfilter_nominal.root"
        #input_file_path = "/eos/experiment/fcc/users/b/brfranco/background_files/sr_photons_kevin/SR_photons_v5/sr_photons_from_positron_182GeVcom_halo_v23_mediumfilter_noSRMask.root"

        input_file_path = "./100MeVelectron_IDEAo1_v03.root"

        #input_file_path = "/eos/experiment/fcc/users/b/brfranco/background_files/sr_photons_kevin/SR_photons_v5/sr_photons_from_positron_182GeVcom_halo_v23_mediumfilter_nominal.root"
        #input_file_path = "/eos/experiment/fcc/users/b/brfranco/background_files/sr_photons_kevin/SR_photons_v5/sr_photons_from_positron_182GeVcom_halo_v23_mediumfilter_halved_nominal.root"
        #input_file_path = "/eos/experiment/fcc/users/b/brfranco/background_files/sr_photons_kevin/SR_photons_v5/sr_photons_from_20Mpositron_45GeVcom_halo_v23_mediumfilter_nominal.root"
        print("\tTreating: %s"%input_file_path)
        podio_reader = root_io.Reader(input_file_path)
        metadata = podio_reader.get("metadata")[0]
        cellid_encoding = metadata.get_parameter("DCHCollection__CellIDEncoding")
        decoder = dd4hep.BitFieldCoder(cellid_encoding)
        for event in podio_reader.get("events"):
            bx_seen += 1 # now there is one event per rootfile representing 1 bx, may change in future so place the counter here
            # loop over MCParticles
            for particle in event.get("MCParticles"):
                particle_fourvector = ROOT.Math.LorentzVector('ROOT::Math::PxPyPzM4D<double>')(particle.getMomentum().x, particle.getMomentum().y, particle.getMomentum().z, particle.getMass())
                if abs(particle.getPDG()) == 11:
                    pt_all_electrons.Fill(particle_fourvector.pt())
                    if (particle.getGeneratorStatus() >= 1): # 1 or more = primary, 0 = secondary
                        pt_primary_electrons.Fill(particle_fourvector.pt())

            # where are the particles leading to hits in the DC coming from and what energy do they have per bx?
            particle_hittingDC_origin_rz_pt_per_bx = ROOT.TH2F(f"particle_hittingDC_origin_rz_pt_bx_{bx_seen}", f"particle_hittingDC_origin_rz_pt_bx_{bx_seen}", 100, 0, 2500, 100, 0, 2000)
            particle_hittingDC_origin_rz_pt_per_bx.SetTitle(f"Origin of bkg particles hitting the DCH ({number_of_iteration_on_bx_batches * number_of_bx} BXs)")
            particle_hittingDC_origin_rz_pt_per_bx.GetXaxis().SetTitle("z [mm]")
            particle_hittingDC_origin_rz_pt_per_bx.GetYaxis().SetTitle("r [mm]")
            particle_hittingDC_origin_rz_pt_per_bx.GetZaxis().SetTitle("P_{T} [GeV]")
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
            for dc_hit in event.get("DCHCollection"):
                total_number_of_hit += 1
                total_number_of_hit_integrated_per_batch += 1
                total_number_of_hit_thisbx += 1
                cellID = dc_hit.getCellID()
                layer = decoder.get(cellID, "layer")
                superlayer = decoder.get(cellID, "superlayer")
                nphi = decoder.get(cellID, "nphi")
                particle = dc_hit.getParticle()
                # are the hit from the DC coming from a primary particle or not? # w.r.t. to isPrimary_of_particles_hitting_dch, this one is filled once per hit
                isPrimary_of_particle_attached_to_hits.Fill(int(particle.getGeneratorStatus() and not dc_hit.isProducedBySecondary()))
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
                # Where do the particles hit the DC?
                DC_simhit_position_rz.Fill(abs(dc_hit.getPosition().z), sqrt(dc_hit.getPosition().x ** 2 + dc_hit.getPosition().y ** 2))
                DC_simhit_position_xy.Fill(dc_hit.getPosition().x, dc_hit.getPosition().y)
                # Map of the fired cells energies
                DC_fired_cell_map.Fill(nphi, unique_layer_index, 1e+3*dc_hit.getEDep())
                DC_fired_cell_map_per_evt.Fill(nphi, unique_layer_index, 1e+3*dc_hit.getEDep())
                # what is the de/DX of the background particles/muons (from another rootfile)?
                if not dc_hit.getPathLength() == 0:
                    dedx_of_dch_hits.Fill(1e+6 * dc_hit.getEDep()/dc_hit.getPathLength())
                # fill the particle related TH1's (only once, a particle can lead to multiple DC hits)
                if particle.getObjectID().index not in seen_particle_ids:
                    particle_fourvector = ROOT.Math.LorentzVector('ROOT::Math::PxPyPzM4D<double>')(particle.getMomentum().x, particle.getMomentum().y, particle.getMomentum().z, particle.getMass())
                    total_number_of_particles_reaching_dch_per_bx_batch += 1
                    # where are the particles leading to hits in the DC coming from?
                    particle_origin = particle.getVertex()
                    particle_hittingDC_origin_rz.Fill(abs(particle_origin.z), sqrt(particle_origin.x ** 2 + particle_origin.y ** 2))
                    particle_hittingDC_origin_rz_pt_per_bx.Fill(abs(particle_origin.z), sqrt(particle_origin.x ** 2 + particle_origin.y ** 2), particle_fourvector.pt())
                    particle_hittingDC_origin_xy.Fill(particle_origin.x, particle_origin.y)
                    # what is the energy of the particles actually hitting the DC
                    energy_of_particles_hitting_dch.Fill(particle_fourvector.E())
                    energy_of_particles_hitting_dch_lowValues.Fill(particle_fourvector.E())
                    # what is the pt of the particles actually hitting the DC and with vertex radius below 10 cm
                    if(sqrt(particle_origin.x ** 2 + particle_origin.y ** 2) < 100):
                        pt_of_particles_hitting_dch_below_10cm.Fill(particle_fourvector.pt())
                    # what is the particle pdgid hitting DC?
                    pdgid_of_particles_hitting_dch.Fill(particle.getPDG())
                    # all particles leaving signals in DCH are electrons or positrons, is their parent a photon?
                    has_photon_parent = 0
                    #print(len(particle.getParents())) usually 0 or 1 for ICP
                    for parent in particle.getParents():
                        if parent.getPDG() == 22:
                            has_photon_parent = 1
                            energy_of_particles_hitting_dch_with_photon_parent.Fill(particle_fourvector.E())
                    hasPhotonParent_of_particles_hitting_dch.Fill(has_photon_parent)
                    hasPhotonParentOrIsPhoton_of_particles_hitting_dch.Fill(int(has_photon_parent or particle.getPDG() == 22))

                    # are the particles hitting dc primaries or secondaries?
                    isPrimary_of_particles_hitting_dch.Fill((int(particle.getGeneratorStatus() and not dc_hit.isProducedBySecondary())))
                    seen_particle_ids.append(particle.getObjectID().index) # must be at the end

                # what is the vertex radius of the oldest parent (original primary particle)?
                # put this one outside of the seen_particle_ids condition to "weight" with the number of hit the particle lead to
                is_oldest_parent = False
                current_parent = particle
                while not is_oldest_parent:
                    if current_parent.parents_size() != 0:
                        current_parent = current_parent.getParents(0)
                    else:
                        is_oldest_parent = True
                # what is the vertex radius of the oldest parent (original primary particle)?
                parent_origin = current_parent.getVertex()
                parent_vertex_radius = sqrt(parent_origin.x ** 2 + parent_origin.y ** 2)
                oldestParentVertexRadius_of_particles_hitting_dch.Fill(parent_vertex_radius)
                if parent_vertex_radius > 10:
                    total_number_of_hit_comingFromPrimaryAfterBeamPipe += 1

                # end of loop on sim hits
            for particleKey in dict_particle_n_fired_cell.keys():
                n_cell_fired_of_particles_hitting_dch.Fill(dict_particle_n_fired_cell[particleKey])
                n_cell_fired_of_particles_hitting_dch_log.Fill(dict_particle_n_fired_cell[particleKey])

            print("\t\t\tTotal number of bkg hit in the DC from this BX: ", str(total_number_of_hit_thisbx))
            draw_histo(particle_hittingDC_origin_rz_pt_per_bx, "colz")
            draw_histo(DC_fired_cell_map_per_evt, "colz")
            # end of loop on events (there is 1 event per BX so it is equivalent to the loop on BXs)

        percentage_of_fired_cells = 100 * len(dict_cellID_nHits.keys())/float(total_number_of_cells)
        print("\t\tTotal number of bkg hit in the DC accumulating BXs of one batch: ", str(total_number_of_hit_integrated_per_batch))
        print("\t\tNumber of fired cells: ", str(len(dict_cellID_nHits.keys())))
        print("\t\tNumber of cells with more than one hit: ", str(number_of_cell_with_multiple_hits))
        print("\t\tPercentage of fired cells: ", percentage_of_fired_cells, " %")
        print("\t\tTotal number of particles so far: ", total_number_of_particles_per_bx_batch)
        print("\t\tTotal number of particles that have hit the DC so far: ", total_number_of_particles_reaching_dch_per_bx_batch)
        print("\t\tPercentage of particles reaching the DC so far: ", 100 * total_number_of_particles_reaching_dch_per_bx_batch / float(total_number_of_particles_per_bx_batch), " %")

        # end of loop on BX files

    if len(bx_files_seen_so_far) < number_of_bx:
        print(f"\033[91mThere were not enough files left to complete iteration {bx_batch_index}: {number_of_bx} required, {len(bx_files_seen_so_far)} available. Stopping here.\033[0m")
        break

    string_for_overall_occupancies += f"BX batch iteration {bx_batch_index}, integrating over {number_of_bx} BXs\n"
    string_for_overall_occupancies += f"\tTotal number of bkg hit in the DC accumulating BXs of one batch: {total_number_of_hit_integrated_per_batch}\n"
    string_for_overall_occupancies += f"\tNumber of fired cells: {len(dict_cellID_nHits.keys())}\n"
    string_for_overall_occupancies += f"\tPercentage of fired cells: {len(dict_cellID_nHits.keys())/float(total_number_of_cells)}\n"

    # Write the "per batch" histograms that have to be integrated over the BXs inside a single batch
    draw_histo(DC_simhit_position_rz, "colz")
    draw_histo(DC_simhit_position_xy, "colz")
    draw_histo(DC_fired_cell_map, "colz")
    # Normalize the occupancy per layer th1 (divide the number of cell fired by the total number of cell) and fill the TProfile of occupancies
    for unique_layer_index in range(0, total_number_of_layers):
        raw_bin_content = occupancy_per_layer.GetBinContent(unique_layer_index + 1) # NB: we use the trick that the bin index here is the same as the layer index it corresponds to, just binIdx 0 is underflow
        occupancy_per_layer.SetBinContent(unique_layer_index + 1, 100 * raw_bin_content/float(n_cell_per_layer[str(unique_layer_index)])) # unique_layer_index and n_cell_per_layer key definitions coincide
        occupancy_per_layer_profile.Fill(unique_layer_index, 100 * raw_bin_content/float(n_cell_per_layer[str(unique_layer_index)]))
    draw_histo(occupancy_per_layer)
    overall_occupancies.append(percentage_of_fired_cells)

#print(f"Percentage of hit coming from primary particles with vertex radius after beampipe: {total_number_of_hit_comingFromPrimaryAfterBeamPipe / float(total_number_of_hit)}")
# end of loop on BX batches
draw_histo(particle_hittingDC_origin_rz, "colz")
draw_histo(particle_hittingDC_origin_xy, "colz")
draw_histo(particle_hittingDC_origin_rz, "colzlogz")
draw_histo(dedx_of_dch_hits)
draw_histo(energy_of_particles_hitting_dch, logY = True)
draw_histo(energy_of_particles_hitting_dch_lowValues, logY = True)
draw_histo(energy_of_particles_hitting_dch_with_photon_parent, logY = True)
pdgid_of_particles_hitting_dch.Scale(1/pdgid_of_particles_hitting_dch.Integral())
draw_histo(pdgid_of_particles_hitting_dch)
draw_histo(hasPhotonParent_of_particles_hitting_dch)
draw_histo(hasPhotonParentOrIsPhoton_of_particles_hitting_dch)
isPrimary_of_particles_hitting_dch.Scale(1/isPrimary_of_particles_hitting_dch.Integral())
draw_histo(isPrimary_of_particles_hitting_dch)
isPrimary_of_particle_attached_to_hits.Scale(1/isPrimary_of_particle_attached_to_hits.Integral())
draw_histo(isPrimary_of_particle_attached_to_hits)
draw_histo(oldestParentVertexRadius_of_particles_hitting_dch)
draw_histo(n_cell_fired_of_particles_hitting_dch)
draw_histo(n_cell_fired_of_particles_hitting_dch_log, logY = True)
draw_histo(pt_of_particles_hitting_dch_below_10cm)
draw_histo(occupancy_per_layer_profile)
# MCParticle plots
draw_histo(pt_all_electrons)
draw_histo(pt_all_electrons, logY = True)
draw_histo(pt_primary_electrons)
draw_histo(pt_primary_electrons, logY = True)
print(f"Number of electrons potentially reaching (includes secondaries with larger radius --> under estimated) DCH for {number_of_iteration_on_bx_batches * number_of_bx} BX's: ", pt_all_electrons.Integral(pt_all_electrons.FindBin(0.35), pt_all_electrons.GetNbinsX()))

print("Overall occupancies for all iterations: ", str(overall_occupancies))
with open(os.path.join(plot_folder, "overall_occupancies.txt"), "w") as myfile:
    myfile.write(string_for_overall_occupancies)
    myfile.write("Overall occupancies for each iteration: " + str(overall_occupancies))
print(os.path.join(plot_folder, "overall_occupancies.txt"), " written.")

# cross check the number of cell
#for layer in dict_layer_phiSet.keys():
#    print(f"Number of phi cells in layer {layer}")
#    print(len(dict_layer_phiSet[layer]))
#    print(dict_layer_phiSet[layer])

# draw averaged quantities
overall_occupancies_th1 = ROOT.TH1F(f"overall_occupancies_th1", f"overall_occupancies_th1", 50, 0, 5)
overall_occupancies_th1.SetTitle(f"DC overall occupancy, {number_of_bx} BXs ran {number_of_iteration_on_bx_batches} times")
overall_occupancies_th1.GetXaxis().SetTitle("Overall occupancy [%]")
overall_occupancies_th1.GetYaxis().SetTitle("Number of entries")
for occupancy in overall_occupancies:
    overall_occupancies_th1.Fill(occupancy)
draw_histo(overall_occupancies_th1)
