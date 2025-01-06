## Background studies

### IDEA 
#### Data
Directory: `/eos/experiment/fcc/ee/datasets/DC_tracking/Pythia/scratch/Zcard_CLD_background_v1/`
The directory contains 4000 folders (corresponding to different seeds), each folder contains a rootfile with `out_sim_edm4hep.root` which contains the result of the overlay hits in the collections "New..", and `output_IDEA_DIGI.root` which contains the output of the digitizer.

Background only sim files: `/eos/experiment/fcc/ee/datasets/DC_tracking/Pythia/IDEA_background_only/out_sim_edm4hep_background_*.root`
#### Visualization
To visualize the detector with the hits resulting from the physics event and beam background source the key4hep nightlies and use the `edm4hep2json` command on one of the `out_sim_edm4hep.root` (result of the overlay) and use [Phoenix](https://fccsw.web.cern.ch/fccsw/phoenix-dev/) to load the file and explore the hits.

### CLD

#### Data
Directory: `/eos/experiment/fcc/ee/datasets/CLD_tracking/condor/Pythia/CLD_background_overlay/`
The directory contains 4000 folders (corresponding to different seeds), each folder contains a rootfile with `output_overlay_new1.root` which contains the result of the overlay hits in the collections "New..", and `out_reco_edm4hep_REC.edm4hep.root` which contains the result of the reconstruction algorithm of CLD.

#### Visualization
To visualize the detector with the hits resulting from the physics event and beam background dowload this file: `/eos/experiment/fcc/ee/datasets/CLD_tracking/condor/Pythia/CLD_background_overlay/10/output_overlay_new1.edm4hep.json` and use [Phoenix](https://fccsw.web.cern.ch/fccsw/phoenix-dev/) to load the file. If you want to crease a new file from an edm4hep sim file, source the key4hep nightlies and use the `edm4hep2json` command.