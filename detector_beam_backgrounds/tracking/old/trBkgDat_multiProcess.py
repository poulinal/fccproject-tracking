import numpy as np
import math
from podio import root_io
from ROOT import dd4hep
from concurrent.futures import ProcessPoolExecutor, as_completed

# Constants and file paths
oldDataPath = "/ceph/submit/data/group/fcc/ee/detector/tracking/IDEA_background_only/"
newDataPath = "/ceph/submit/data/group/fcc/ee/detector/tracking/IDEA_background_only_IDEA_o1_v03_v3/"

numfiles = 500
list_overlay = [
    newDataPath + f"out_sim_edm4hep_background_{i}.root"
    for i in range(1, numfiles + 1)
]

keys = [
    "R", "p", "px", "py", "pz", "gens", "pos_ver", "hits", "hits_pL",
    "pos_hit", "unique_mcs", "superLayer", "layer", "nphi", "stereo",
    "pos_z", "count_hits", "has_par_photon", "pdg",
    "hits_produced_secondary", "hits_mc_produced_secondary", "percentage_of_fired_cells"
]

dic_file_path = "fccproject-tracking/detector_beam_backgrounds/tracking/data/new_background_particles_500.npy"

# Initialize the dictionary with empty lists
dic = {key: [] for key in keys}
np.save(dic_file_path, dic)

# Function to process a single file
def process_file(rootfile):
    # Load existing dictionary
    dic = {key: [] for key in keys}
    
    list_index, list_hit_path_length, list_pos_hit, list_superLayer, list_layer = [], [], [], [], []
    list_nphi, list_pos_z, list_R, list_p, list_px, list_py, list_pz = [], [], [], [], [], [], []
    list_gen_status, list_pos_ver, list_par_photon, list_pdg = [], [], [], []
    list_hits_secondary, list_hits_mc_secondary, count_hits = [], [], []
    seen = []
    
    # Geometry setup
    n_layers_per_superlayer = 8
    n_superlayers = 14
    n_cell_superlayer0 = 192
    n_cell_increment = 48
    n_cell_per_layer = {
        str(n_layers_per_superlayer * sl + l): n_cell_superlayer0 + sl * n_cell_increment
        for sl in range(n_superlayers)
        for l in range(n_layers_per_superlayer)
    }
    total_number_of_cells = sum(n_cell_per_layer.values())
    
    # Read root file
    reader = root_io.Reader(rootfile)
    metadata = reader.get("metadata")[0]
    cellid_encoding = metadata.get_parameter("DCHCollection__CellIDEncoding")
    decoder = dd4hep.BitFieldCoder(cellid_encoding)
    
    # Process events
    for event in reader.get("events"):
        dc_hits = event.get("DCHCollection")
        for dc_hit in dc_hits:
            mcParticle = dc_hit.getMCParticle()
            index_mc = mcParticle.getObjectID().index
            
            list_index.append(index_mc)
            list_pos_hit.append([dc_hit.getPosition().x, dc_hit.getPosition().y, dc_hit.getPosition().z])
            list_pos_z.append(dc_hit.getPosition().z)
            
            if index_mc not in seen:
                count_hits.append(1)
                list_hits_mc_secondary.append(dc_hit.isProducedBySecondary())
                seen.append(index_mc)
            else:
                index = seen.index(index_mc)
                count_hits[index] += 1
                list_hits_mc_secondary[index] = max(list_hits_mc_secondary[index], dc_hit.isProducedBySecondary())
            
            list_hits_secondary.append(dc_hit.isProducedBySecondary())
            
            cellID = dc_hit.getCellID()
            superlayer = decoder.get(cellID, "superlayer")
            layer = decoder.get(cellID, "layer")
            nphi = decoder.get(cellID, "nphi")
            list_superLayer.append(superlayer)
            list_layer.append(layer)
            list_nphi.append(nphi)
    
    # Add particle-level information
    MCparticles = reader.get("events")[0].get("MCParticles")
    unique_mcs = np.unique(list_index)
    for mc_index in unique_mcs:
        mcParticle = MCparticles[int(mc_index)]
        x_vertex = mcParticle.getVertex().x
        y_vertex = mcParticle.getVertex().y
        z_vertex = mcParticle.getVertex().z
        vertex_R = math.sqrt(mcParticle.getVertex().x**2 + mcParticle.getVertex().y**2) * 1e-03
        list_R.append(vertex_R)
        momentum = mcParticle.getMomentum()
        p = math.sqrt(momentum.x**2 + momentum.y**2 + momentum.z**2)
        list_p.append(p)
        list_px.append(momentum.x)
        list_py.append(momentum.y)
        list_pz.append(momentum.z)
        list_gen_status.append(mcParticle.getGeneratorStatus())
        list_pdg.append(mcParticle.getPDG())
        
        has_photon_parent = 0
        for parent in mcParticle.getParents():
            if parent.getPDG() == 22:
                has_photon_parent = 1
        list_par_photon.append(has_photon_parent)
    
    # Update dictionary with processed data
    dic["R"].extend(list_R)
    dic["p"].extend(list_p)
    dic["px"].extend(list_px)
    dic["py"].extend(list_py)
    dic["pz"].extend(list_pz)
    dic["gens"].extend(list_gen_status)
    dic["pos_ver"].extend(list_pos_ver)
    dic["hits"].extend(list_index)
    dic["pos_hit"].extend(list_pos_hit)
    dic["unique_mcs"].extend(unique_mcs)
    dic["superLayer"].extend(list_superLayer)
    dic["layer"].extend(list_layer)
    dic["nphi"].extend(list_nphi)
    dic["pos_z"].extend(list_pos_z)
    dic["count_hits"].extend(count_hits)
    dic["has_par_photon"].extend(list_par_photon)
    dic["pdg"].extend(list_pdg)
    dic["hits_produced_secondary"].extend(list_hits_secondary)
    dic["hits_mc_produced_secondary"].extend(list_hits_mc_secondary)
    dic["percentage_of_fired_cells"].append(
        100 * len(set(list_index)) / total_number_of_cells
    )
    
    return dic

# Function to combine results
def combine_results(results, output_path):
    combined_dic = {key: [] for key in keys}
    for dic in results:
        for key in keys:
            combined_dic[key].extend(dic[key])
    np.save(output_path, combined_dic)

# Main parallel processing
if __name__ == "__main__":
    results = []
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file): file for file in list_overlay}
        for future in as_completed(future_to_file):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing file {future_to_file[future]}: {e}")
    combine_results(results, dic_file_path)
