from podio import root_io
list_overlay = []

rootfile = '/ceph/submit/data/group/fcc/ee/detector/tracking/IDEA_background_only/out_sim_edm4hep_background_1.root'
# rootfile = '/ceph/submit/data/group/fcc/ee/detector/VTXStudiesFullSim/CLD_guineaPig_andrea_June2024_v23/pairs_1.root'
print(rootfile)



reader = root_io.Reader(rootfile)
events = reader.get("events")


print("before loop")
for event in events:
    print("before dc_hits")
#     dc_hits = event.get("VertexBarrelCollection")
#     print("after dc_hits")


# source /work/submit/jaeyserm/software/FCCAnalyses/setup.sh

# source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2024-10-03
# /work/submit/poulin/fccproject-tracking/detector_beam_backgrounds/tracking/example_script_explore_background_hits_DC.py
# /work/submit/poulin/fccproject-tracking/detector_beam_backgrounds/tracking/example_original.py