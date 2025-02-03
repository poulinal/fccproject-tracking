

def getOccupancy():
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
    list_overall_occupancy = []
    #total_number_of_hit_comingFromPrimaryAfterBeamPipe = 0 
    
    # cross check the number of cell
    dict_layer_phiSet = {}
    
    
    
    #loop
    
    dict_cellID_nHits = {}
    number_of_cell_with_multiple_hits = 0
    
    list_occupancy_per_layer = []
    
    # what fraction of particles do reach the drift chamber?
    # total_number_of_particles_reaching_dch_per_bx_batch = 0
    # total_number_of_particles_per_bx_batch = 0
    
    
    
    
    #before hit loop
    # total_number_of_hit_thisbx = 0
    #seen_particle_ids = []
    dict_particle_n_fired_cell = {}
    dict_particle_fired_cell_id = {}
    total_number_of_particles_per_bx_batch += len(event.get("MCParticles"))
    
    
    
    
    
    for dc_hit in event.get("DCHCollection"):
        cellID = dc_hit.getCellID()
        layer = decoder.get(cellID, "layer")
        superlayer = decoder.get(cellID, "superlayer")
        nphi = decoder.get(cellID, "nphi")
        particle = dc_hit.getParticle()
        
        # are the hit from the DC coming from a primary particle or not? # w.r.t. to isPrimary_of_particles_hitting_dch, this one is filled once per hit
        #isPrimary_of_particle_attached_to_hits.Fill(int(particle.getGeneratorStatus() and not dc_hit.isProducedBySecondary()))
        
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
            list_occupancy_per_layer.Fill(unique_layer_index)
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
        # DC_fired_cell_map.Fill(nphi, unique_layer_index, 1e+3*dc_hit.getEDep())
        # DC_fired_cell_map_per_evt.Fill(nphi, unique_layer_index, 1e+3*dc_hit.getEDep())
        
        # what is the de/DX of the background particles/muons (from another rootfile)?
        # if not dc_hit.getPathLength() == 0:
        #     dedx_of_dch_hits.Fill(1e+6 * dc_hit.getEDep()/dc_hit.getPathLength())

        '''
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
        '''
        
        # end of loop on sim hits
    for particleKey in dict_particle_n_fired_cell.keys():
        # n_cell_fired_of_particles_hitting_dch.Fill(dict_particle_n_fired_cell[particleKey])
        # n_cell_fired_of_particles_hitting_dch_log.Fill(dict_particle_n_fired_cell[particleKey])
        list_n_cells_fired_mc.append(dict_particle_n_fired_cell[particleKey])

    # print("\t\t\tTotal number of bkg hit in the DC from this BX: ", str(total_number_of_hit_thisbx))
    # draw_histo(particle_hittingDC_origin_rz_pt_per_bx, "colz")
    # draw_histo(DC_fired_cell_map_per_evt, "colz")
    # # end of loop on events (there is 1 event per BX so it is equivalent to the loop on BXs)




    percentage_of_fired_cells = 100 * len(dict_cellID_nHits.keys())/float(total_number_of_cells)
    # print("\t\tTotal number of bkg hit in the DC accumulating BXs of one batch: ", str(total_number_of_hit_integrated_per_batch))
    # print("\t\tNumber of fired cells: ", str(len(dict_cellID_nHits.keys())))
    # print("\t\tNumber of cells with more than one hit: ", str(number_of_cell_with_multiple_hits))
    # print("\t\tPercentage of fired cells: ", percentage_of_fired_cells, " %")
    # print("\t\tTotal number of particles so far: ", total_number_of_particles_per_bx_batch)
    # print("\t\tTotal number of particles that have hit the DC so far: ", total_number_of_particles_reaching_dch_per_bx_batch)
    # print("\t\tPercentage of particles reaching the DC so far: ", 100 * total_number_of_particles_reaching_dch_per_bx_batch / float(total_number_of_particles_per_bx_batch), " %")
    
    
    # Normalize the occupancy per layer th1 (divide the number of cell fired by the total number of cell) and fill the TProfile of occupancies
    for unique_layer_index in range(0, total_number_of_layers):
        raw_bin_content = list_occupancy_per_layer.GetBinContent(unique_layer_index + 1) # NB: we use the trick that the bin index here is the same as the layer index it corresponds to, just binIdx 0 is underflow
        list_occupancy_per_layer.SetBinContent(unique_layer_index + 1, 100 * raw_bin_content/float(n_cell_per_layer[str(unique_layer_index)])) # unique_layer_index and n_cell_per_layer key definitions coincide
        occupancies_per_layer.Fill(unique_layer_index, 100 * raw_bin_content/float(n_cell_per_layer[str(unique_layer_index)]))
    list_overall_occupancy.append(percentage_of_fired_cells)
    

