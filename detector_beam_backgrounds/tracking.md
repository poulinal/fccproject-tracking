# Beam Background Studies on Tracker Detectors

This project aims to assess the impact of beam backgrounds on the tracking detectors and the overall tracking performance. At FCC-ee, we are considering two conceptual tracking detector designs: a gaseous drift chamber (light) and a silicon tracker (heavy). The interaction of beam particles with the detector material differs for these designs and must be studied separately.

- The **IDEA detector concept** features a light drift chamber (DC).
- The **CLD detector concept** utilizes a silicon tracker.

## IDEA Detector

### Data
- **Directory:**  
  `/ceph/submit/data/group/fcc/ee/detector/tracking/Zcard_CLD_background_v1/`  
  This directory contains 4000 folders, each corresponding to a different seed. Each folder includes:
  - `out_sim_edm4hep.root`: Contains the overlay hit results in the "New.." collections.
  - `output_IDEA_DIGI.root`: Contains the output of the digitizer.

- **Background-only simulation files:**  
  `/ceph/submit/data/group/fcc/ee/detector/trackinga/IDEA_background_only/out_sim_edm4hep_background_*.root`

### Visualization
To visualize the detector with hits resulting from physics events and beam backgrounds:
1. Source the **key4hep nightlies**. [provide link]
2. Use the `edm4hep2json` command on one of the `out_sim_edm4hep.root` files (result of the overlay).
3. Load the generated JSON file into [Phoenix](https://fccsw.web.cern.ch/fccsw/phoenix-dev/) to explore the hits.

## CLD Detector

### Data
- **Directory:**  
  `/ceph/submit/data/group/fcc/ee/detector/tracking/CLD_background_overlay/`  
  This directory contains 4000 folders, each corresponding to a different seed. Each folder includes:
  - `output_overlay_new1.root`: Contains the overlay hit results in the "New.." collections.
  - `out_reco_edm4hep_REC.edm4hep.root`: Contains the reconstruction algorithm results for CLD.

### Visualization
To visualize the detector with hits from physics events and beam backgrounds:
1. Download the file:  
   `/eos/experiment/fcc/ee/datasets/CLD_tracking/condor/Pythia/CLD_background_overlay/10/output_overlay_new1.edm4hep.json`
2. Load it into [Phoenix](https://fccsw.web.cern.ch/fccsw/phoenix-dev/).
3. To create a new JSON file from an EDM4hep simulation file:
   - Source the **key4hep nightlies**.
   - Use the `edm4hep2json` command to generate the JSON file.
