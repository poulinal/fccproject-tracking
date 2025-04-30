import os
# import ctypes

from podio import root_io
import dd4hep as dd4hepModule
from ROOT import dd4hep
import ROOT
from ROOT import TMath, TVector3

import build.dch_module as dch_module


class DCH_info(dch_module.DCH_info):
    def __init__(self, detectorPath=None):
        super().__init__()
        self._database = [] #will be a list of dictionaries
        
        if detectorPath is None:
            RuntimeError("DCH_info: No detector provided")
        else:
            try:
                detector = dd4hep.Detector.getInstance()
                detector.fromXML(detectorPath)
            except RuntimeError as e:
                print(f"Error loading detector: {e}")
                return
        
        dch = self.setup_dch_info(detector)
        
        # Check if database is valid
        if dch.IsDatabaseEmpty():
            print("Error: DCH database is not valid / is empty")
            return
        
        # # Example: Calculate wire position for a specific layer and cell
        # ilayer = 10
        # nphi = 5
        
        # # Get wire direction vector (returned as Python tuple)
        # wire_vec = dch.Calculate_wire_vector_ez(ilayer, nphi)
        # print(f"Wire direction vector for layer {ilayer}, cell {nphi}: {wire_vec}")
        
        # # Calculate point at z=0 for this wire
        # wire_z0 = dch.Calculate_wire_z0_point(ilayer, nphi)
        # print(f"Wire z0 point: {wire_z0}")
        
        # # Calculate distance from a hit to wire
        # hit_pos = (500.0, 500.0, 100.0)  # Example hit position in mm
        # hit_to_wire = dch.Calculate_hitpos_to_wire_vector(ilayer, nphi, hit_pos)
        # distance = (hit_to_wire[0]**2 + hit_to_wire[1]**2 + hit_to_wire[2]**2)**0.5
        # print(f"Distance from hit to wire: {distance} mm")
        
        #access the database
        self._database = dch.get_database_as_list_dic()
        print(self._database) #radius_sw_z0 is the radius of the layer at z=0

    def get_database_as_list_dic(self):
        return self._database

    def Set_database(self, database):
        self._database = database

    def IsDatabaseEmpty(self):
        return len(self._database) == 0

    def setup_dch_info(self, detector):
        # Create a DCH_info object
        dch_info = dch_module.DCH_info()
        
        # print(detector.constantAsDouble("DCH_gas_inner_cyl_R"))
        # dch_info.Set_rin(detector.constantAsDouble("DCH_gas_inner_cyl_R") * dch_module.mm)
        dch_info.Set_rin(detector.constantAsDouble("DCH_gas_inner_cyl_R"))
        dch_info.Set_rout(detector.constantAsDouble("DCH_gas_outer_cyl_R"))
        dch_info.Set_lhalf(detector.constantAsDouble("DCH_gas_Lhalf"))
        
        dch_info.Set_guard_rin_at_z0(detector.constantAsDouble("DCH_guard_inner_r_at_z0"))
        dch_info.Set_guard_rout_at_zL2(detector.constantAsDouble("DCH_guard_outer_r_at_zL2"))
        
        dch_info.Set_ncell0(detector.constantAsLong("DCH_ncell"))
        dch_info.Set_ncell_increment(detector.constantAsLong("DCH_ncell_increment"))
        dch_info.Set_ncell_per_sector(detector.constantAsLong("DCH_ncell_per_sector"))
        
        dch_info.Set_nlayersPerSuperlayer(detector.constantAsLong("DCH_nlayersPerSuperlayer"))
        dch_info.Set_nsuperlayers(detector.constantAsLong("DCH_nsuperlayers"))
        
        dch_info.Set_first_width(detector.constantAsDouble("DCH_first_width"))
        dch_info.Set_first_sense_r(detector.constantAsDouble("DCH_first_sense_r"))
        
        dch_info.Set_twist_angle(2 * detector.constantAsDouble("DCH_alpha"))
        
        # Build the layer database based on parameters
        dch_info.BuildLayerDatabase(verbose=True)
        
        return dch_info

