# Impact of beam backgrounds on the vertex detector

In this project, we will characterize the beam background and physics processes on the vertex detector. The vertex detector consists of small cylindrical layers of silicon pixels that get activated when a charged particle traverses the silicon layer. The signals of all the layers are then read by the readout system and will be used to reconstruct the particle's trajectory and determine its momentum by bending through the magnetic field.


We simulated in detail the interaction of charged particles with the detector materials, which is done in the program called Geant4. In this program, the layout of the vertex detector is properly defined, and each particle (either signal or background), is transported to the detector and the interaction with all the materials is calculated.

Presentation of Nate: https://github.com/mit-fcc/projects/blob/master/detector_beam_backgrounds/vertexing/NateMartinez_16102024.pdf

The scripts are located in the ```scripts``` directory (either fork or copy them over to subMIT). As always, before you execute them, run the following command:

```
source /work/submit/jaeyserm/software/FCCAnalyses/setup.sh
```


## Energy deposits

Each particle (whether signal or background) leaves a hit in the vertex detector layers when it crosses the layer. It's a small energy deposit in the order of keV. To compute the energy deposit in each layer, execute the following command:

```
python energy_deposit.py --calculate --maxFiles 100
```

In the ```energy_deposit.py```, you can specify your detector (IDEA or CLD) and the process (beam backgrounds or physics events, Z->hadrons in this case). The ```maxFiles``` argument is optional (remove it if you want to run over all the files and events). 


To plot the energy distributions for each layer, execute the following command:

```
python energy_deposit.py --plots
```

It generates the energy deposits for each layer. There are 3 types of energy deposit plots:

- Raw energy deposit for each hit on each layer (keV)
- Energy deposit per unit of crossed material (keV/mm)
- Same as above, but for a subset of hits that do not have secondary particles


## Hit maps


## Occupancy
