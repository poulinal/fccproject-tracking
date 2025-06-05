from trBkgDatUpdateOcc import calcShiftIndicie, globalIndicieShift, stereoWire, check_odd_fractions, fast_check_odd_fractions, calculateOnlyNeighbors
from utilities.utils import faster_check_odd_fractions
import numpy as np
import unittest


class TestCalcShiftIndicie(unittest.TestCase):
    def initialize(self):
        rangeR = 10
        rangeZ = 3 * 100
        rangePhi = 10
        atLeast = 1
        n_cell_per_layer = [4, 4, 4, 8, 8, 8]
        
        '''basic test cases (only r phi)
        base case (all zero):
        [[0 0 0 0],
        [0 0 0 0],
        [0 0 0 0]]
        
        verticle:
        [[0 1 0 0],
        [0 1 0 0],
        [0 1 0 0]]
        
        diagonal:
        [[0 0 0 1],
        [0 0 1 0],
        [0 1 0 0]]
        
        pure horizontal:
        [[0 0 0 0],
        [1 1 1 1],
        [0 0 0 0]]
        
        semi horizontal:
        [[0 1 0 0],
        [1 0 1 1],
        [0 0 0 0]]
        '''
        z = 1
        t_r = (0, 0, z) #wont ever need this for testing calcShiftIndicie
        self.n_cell_per_layer = [4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8]
        arc_length_shift = 45
        # arc_length_step = 2 * np.pi * z / n_cell_per_layer[str(0)]
        # z_layer_to_shift = {0: [0, 0, t_r, arc_length_shift, arc_length_step]}
        #key is z, list of tuples of (layer, globalRadius, (x_new, y_new, z), arc_length_shift, arc_length_step)
        self.placeholder_t_r = (0, 0, 0) #not used in calcShiftIndicie, but needed for the function signature
        
        #takes parent of (layer, globalRadius, (x_new, y_new, z), arc_length_shift, arc_length_step) and
        
        '''across z test cases
        
        '''
    
    def test_basic(self):
        
        self.initialize()
        parentinfo = (0, 0, self.placeholder_t_r, 0, 100)
        neighborinfo = (0, 0, self.placeholder_t_r, 0, 100)
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), 0) #no shift should mean no shift in indicie
        
        parentinfo = (0, 0, self.placeholder_t_r, 0, 100)
        neighborinfo = (0, 0, self.placeholder_t_r, -30, 100)
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), 0) #same layer so no shift in indicie
        
        parentinfo = (0, 0, self.placeholder_t_r, 0, 100)
        neighborinfo = (1, 0, self.placeholder_t_r, -30, 100)
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), 0) #only 1/4Sp so no shift in indicie
        
        parentinfo = (0, 0, self.placeholder_t_r, 0, 100)
        neighborinfo = (1, 0, self.placeholder_t_r, -55, 100)
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), -1) #forward shift in z, so shift in indicie like normal
        
        parentinfo = (0, 0, self.placeholder_t_r, 0, 100)
        neighborinfo = (2, 0, self.placeholder_t_r, 55, 100)
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), 1) #same sign so no shift in indicie
        
    # @unittest.skip('Work in progress')
    def test_across_superlayer(self):
        self.initialize()
        parentinfo = (0, 0, self.placeholder_t_r, 0, 100)
        neighborinfo = (8, 0, self.placeholder_t_r, -50, 100)
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), -1) #across superlayer so shift in indicie
        
        parentinfo = (0, 0, self.placeholder_t_r, 0, 100)
        neighborinfo = (8, 0, self.placeholder_t_r, -80, 120)
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), -1) #across superlayer (update step) so shift in indicie
        
        parentinfo = (0, 0, self.placeholder_t_r, 0, 100)
        neighborinfo = (8, 0, self.placeholder_t_r, -50, 150)
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), 0) #across superlayer (update step) bbut new step too much so no shift in indicie; in reality the step would become smaller with a larger superlayer
        
        parentinfo = (0, 0, self.placeholder_t_r, 0, 100)
        neighborinfo = (8, 0, self.placeholder_t_r, -50, 80)
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), -1) #across superlayer (update step) so  shift in indicie
        
        parentinfo = (0, 0, self.placeholder_t_r, 0, 100)
        neighborinfo = (8, 0, self.placeholder_t_r, -10, 80)
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), 0) #across superlayer (update step)but not enough shift so shift in indicie
        
        parentinfo = (0, 0, self.placeholder_t_r, 10, 100)
        neighborinfo = (9, 0, self.placeholder_t_r, 50, 100)
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), 1) #across superlayer but same stereo
    
    # @unittest.skip('Work in progress')
    def test_parent_looking_backwards(self):
        self.initialize()
        
        parentinfo = (2, 0, self.placeholder_t_r, 0, 100)
        neighborinfo = (1, 0, self.placeholder_t_r, 0, 100)
        self.assertEqual(calcShiftIndicie(neighborinfo, parentinfo, self.n_cell_per_layer), 0) #basic no shift
        
        parentinfo = (2, 0, self.placeholder_t_r, 0, 100)
        neighborinfo = (1, 0, self.placeholder_t_r, -50, 100)
        self.assertEqual(calcShiftIndicie(neighborinfo, parentinfo, self.n_cell_per_layer), 1) #shift should be the same as if parent was looking forward
        
        parentinfo = (3, 0, self.placeholder_t_r, 0, 100)
        neighborinfo = (1, 0, self.placeholder_t_r, -50, 100)
        self.assertEqual(calcShiftIndicie(neighborinfo, parentinfo, self.n_cell_per_layer), 1) #across superlayer shift but treated same as step is same
        
        parentinfo = (3, 0, self.placeholder_t_r, 0, 150)
        neighborinfo = (1, 0, self.placeholder_t_r, -50, 100)
        self.assertEqual(calcShiftIndicie(neighborinfo, parentinfo, self.n_cell_per_layer), 1) #across superlayer shift should act like normal since the neighbor layer will shift 1 step back and another cell will be closer
        
        parentinfo = (3, 0, self.placeholder_t_r, 0, 40)
        neighborinfo = (1, 0, self.placeholder_t_r, -50, 100)
        self.assertEqual(calcShiftIndicie(neighborinfo, parentinfo, self.n_cell_per_layer), 1) #while the parent has small steps, it still really only depends on the neighbor step as that shift will determine which cell is closer... doesnt really matter what the short step next to the parent is
    
    
    def test_parent_non_zero_shift(self):
        self.initialize()
        
        parentinfo = (0, 0, self.placeholder_t_r, 30, 100)
        neighborinfo = (1, 0, self.placeholder_t_r, 0, 100)
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), 0) #parent has a shift so effective neighbor should shift back (except only 1/4Sp for parent)
        
        parentinfo = (0, 0, self.placeholder_t_r, 50, 100)
        neighborinfo = (1, 0, self.placeholder_t_r, 0, 100)
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), -1) #parent has a shift so effective neighbor should shift back 
        
        parentinfo = (0, 0, self.placeholder_t_r, 30, 100)
        neighborinfo = (1, 0, self.placeholder_t_r, -30, 100)
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), -1) #both shift
        
        parentinfo = (0, 0, self.placeholder_t_r, 50, 100)
        neighborinfo = (1, 0, self.placeholder_t_r, -30, 100)
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), -1) #both shift some
        
        parentinfo = (0, 0, self.placeholder_t_r, 50, 100)
        neighborinfo = (1, 0, self.placeholder_t_r, -50, 100)
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), -2) #both shift
        
        parentinfo = (0, 0, self.placeholder_t_r, 50, 100)
        neighborinfo = (1, 0, self.placeholder_t_r, 50, 100)
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), 0) #cant shift since same stereo
        
        parentinfo = (0, 0, self.placeholder_t_r, 40, 80) #shifts left 1 full indicie
        neighborinfo = (8, 0, self.placeholder_t_r, -60, 40) #shifts right 3 full indicie
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), -5) #both shift across superlayer (parent search out)
        
        parentinfo = (8, 0, self.placeholder_t_r, 40, 40)#shifts 1 full indicie
        neighborinfo = (0, 0, self.placeholder_t_r, -60, 80) #shifts 3 full indicie
        self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), -3) #both shift across superlayer (parent search backwards)
        
        
        # parentinfo = (8, 0, self.placeholder_t_r, 45, 40)#shifts 1 full indicie
        # neighborinfo = (0, 0, self.placeholder_t_r, -60, 80) #shifts 3 full indicie
        # self.assertEqual(calcShiftIndicie(parentinfo, neighborinfo, self.n_cell_per_layer), -4) #both shift across superlayer (parent search backwards)
        #Todo, technically this should be -4 but approximation means its -3
        
        
class TestGlobalIndicieShift(unittest.TestCase):
    def initialize(self):
        rangeR = 10
        rangeZ = 3 * 100
        rangePhi = 10
        atLeast = 1
        n_cell_per_layer = [4, 4, 4, 8, 8, 8]
        
        self.z = 1
        self.t_r = (0, 0, self.z) #wont ever need this for testing calcShiftIndicie
        # n_cell_per_layer = 4
        self.arc_length_shift = 45
        self.arc_length_step = 2 * np.pi * self.z / n_cell_per_layer[0]
        # print(f"arc_lnegth_step (1,0,1): {2 * np.pi * 1 / n_cell_per_layer[1]}")
        # self.z_layer_to_shift = {0: [(0, 0, t_r, 0, arc_length_step)]}
        #key is z, list of tuples of (layer, globalRadius, (x_new, y_new, z), arc_length_shift, arc_length_step)
        
    # @unittest.skip('skip since change from every odd 1/4Sp to every 1/4Sp')
    def test_global_indicie_shift(self):
        """Test the globalIndicieShift function with a basic case. We basically want to tests that the GIS function really only depends on arclength shift and step"""
        self.initialize()
        # Example test, replace with your own cases
        # input_data = [1, 2, 3]
        z_layer_to_shift = {0: [(0, 0, self.t_r, 0, 100), (1, 0, self.t_r, -30, 100), (2, 0, self.t_r, 55, 100)], 1: [(0, 0, self.t_r, 0, 100), (1, 0, self.t_r, -50, 100), (2, 0, self.t_r, 150, 100)]}
        #key is z, list of tuples of (layer, globalRadius, (x_new, y_new, z), arc_length_shift, arc_length_step)
        input_data = z_layer_to_shift
        # shift = 1
        # expected = {(0, 0): (False, 0), (1,0): (False, 0), (2,0): (True, 1), (0, 1): (False, 0), (1, 1): (True, -1), (2,1): (True, 3)} #(layer, z): shifted_indicie
        expected = {(0, 0): 0, (1,0): 0, (2,0): 1, (0, 1): 0, (1, 1): -1, (2,1): 3} #(layer, z): shifted_indicie
        #note result of (1,0) is -0.5 but global indicie shift rounds towards zero so it becomes 0 (since if we only shift 1/4Sp, doesnt break 1/2Sp)
        result = globalIndicieShift(input_data)
        # print(f"globalIndicieShift result: {result}")
        self.assertEqual(result, expected)
        

class TestCheckOddFractions(unittest.TestCase):
    def test_check_odd_fractions(self):
        # Example test, replace with your own cases
        shift = 0
        constantSp = 1
        expected = 0
        result = check_odd_fractions(0, 1)
        self.assertEqual(check_odd_fractions(0, 1)[1], 0)
        self.assertEqual(check_odd_fractions(0.25, 1)[1], 1)
        self.assertEqual(check_odd_fractions(-0.25, 1)[1], -1)
        self.assertEqual(check_odd_fractions(0.2, 1)[1], 0)
        self.assertEqual(check_odd_fractions(1.25, 1)[1], 5)
        
class TestFastCheckOddFractions(unittest.TestCase):
    def test_fast_check_odd_fractions(self):
        # Example test, replace with your own cases
        shift = 0
        constantSp = 1
        expected = 0
        result = fast_check_odd_fractions(0, 1)
        self.assertEqual(fast_check_odd_fractions(0, 1)[1], 0)
        self.assertEqual(fast_check_odd_fractions(0.25, 1)[1], 1)
        self.assertEqual(fast_check_odd_fractions(-0.25, 1)[1], -1)
        self.assertEqual(fast_check_odd_fractions(0.2, 1)[1], 0)
        self.assertEqual(fast_check_odd_fractions(1.25, 1)[1], 5)
        
        
class TestFasterCheckOddFractions(unittest.TestCase):
    def test_faster_check_odd_fractions(self):
        # Example test, replace with your own cases
        shift = 0
        constantSp = 1
        expected = 0
        result = faster_check_odd_fractions(0, 1)
        self.assertEqual(faster_check_odd_fractions(0, 1)[1], 0)
        self.assertEqual(faster_check_odd_fractions(0.25, 1)[1], 1)
        self.assertEqual(faster_check_odd_fractions(-0.25, 1)[1], -1)
        self.assertEqual(faster_check_odd_fractions(0.2, 1)[1], 0)
        self.assertEqual(faster_check_odd_fractions(1.25, 1)[1], 5)
        
class TestCalculateOnlyNeighbors(unittest.TestCase):
    def initialize(self):
        self.n_cell_per_layer = {0: 4, 1:4, 2:4, 3:4, 4:4, 5:4, 6:4, 7:4, 8:8, 9:8, 10:8}
        self.placeholder_t_r = (0, 0, 0) #not used in calcShiftIndicie, but needed for the function signature
        self.dic_int_vars = {}
        self.dic_int_vars['radiusR'] = 1
        self.dic_int_vars['radiusPhi'] = 1
        self.dic_int_vars['atLeast'] = 1
        self.dic_int_vars['edepRange'] = 2
        self.dic_int_vars['edepAtLeast'] = 2
        self.dic_int_vars['edepLoosen'] = 1
        self.dic_int_vars['zrange'] = 1
        self.dic_int_vars['zStep'] = 1
        
        self.dic_int_vars['NoNeighborsRemoved'] = 0
        self.dic_int_vars['NeighborsRemained'] = 0
        self.dic_int_vars['EdepNeighborsRemained'] = 0
        self.dic_int_vars['NoEdepNeighborsRemoved'] = 0
        
        self.dic_int_vars['list_max_n_cell_per_superlayer'] = [4, 8]
        self.dic_int_vars['list_max_n_cell_per_layer'] = list(self.n_cell_per_layer.values())
        self.dic_int_vars['n_cell_per_layer'] = self.n_cell_per_layer
        
        self.dic_RPhiKey_by_batch = {}
        # self.dic_posToKey_by_batch = {}
        self.dic_list_mcIDs_one_batch = {}
        self.dic_all_occupancies = {}
        self.batchVars = {}
        self.batchVars['pos_only_neighbors'] = []
        self.batchVars['pos_only_neighbors_only_edeps'] = []
        
        
        self.dic_posToKey_by_batch_keys = ["mcID_index", 
                                     "cell_fired_pos", 
                                     "shifted_phi",
                                     "energy_dep_per_cell_RPhiZ",
                                     "energy_dep_per_cell_RPhiZ_noacc",
                                     "energy_dep_per_cell_xyz_noacc",
                                     "pT", "PDG", "prod_sec", "photon_par", "gen_status",
                                     "combined_overlay_status", 
                                    ]
        
        # z_layer_to_shift = self.dic_int_vars['z_layer_to_shift']
        #z : [(layer, globalRadius, (x_new, y_new, z), arc_length_shift, arc_length_step), ...]
        self.dic_int_vars['z_layer_to_shift'] = {
            0: [
                (0, 0, self.placeholder_t_r, 0, 100), 
                (1, 0, self.placeholder_t_r, 0, 100), 
                (2, 0, self.placeholder_t_r, 0, 100),
                (3, 0, self.placeholder_t_r, 0, 100)], 
            1: [
                (0, 0, self.placeholder_t_r, 10, 100), 
                (1, 0, self.placeholder_t_r, -20, 100), 
                (2, 0, self.placeholder_t_r, 30, 100),
                (3, 0, self.placeholder_t_r, -40, 100)],
            2: [
                (0, 0, self.placeholder_t_r, 20, 100),
                (1, 0, self.placeholder_t_r, -30, 100),
                (2, 0, self.placeholder_t_r, 40, 100),
                (3, 0, self.placeholder_t_r, -50, 100)],
            3: [
                (0, 0, self.placeholder_t_r, 30, 100),
                (1, 0, self.placeholder_t_r, -40, 100),
                (2, 0, self.placeholder_t_r, 50, 100),
                (3, 0, self.placeholder_t_r, -60, 100)],
            4: [
                (0, 0, self.placeholder_t_r, 40, 100),
                (1, 0, self.placeholder_t_r, -50, 100),
                (2, 0, self.placeholder_t_r, 60, 100),
                (3, 0, self.placeholder_t_r, -70, 100)],
            5: [
                (0, 0, self.placeholder_t_r, 50, 100),
                (1, 0, self.placeholder_t_r, -60, 100),
                (2, 0, self.placeholder_t_r, 70, 100),
                (3, 0, self.placeholder_t_r, -80, 100)],
            6: [
                (0, 0, self.placeholder_t_r, 60, 100),
                (1, 0, self.placeholder_t_r, -70, 100),
                (2, 0, self.placeholder_t_r, 80, 100),
                (3, 0, self.placeholder_t_r, -90, 100)],
            7: [
                (0, 0, self.placeholder_t_r, 70, 100),
                (1, 0, self.placeholder_t_r, -80, 100),
                (2, 0, self.placeholder_t_r, 90, 100),
                (3, 0, self.placeholder_t_r, -100, 100)],
            8: [
                (0, 0, self.placeholder_t_r, 80, 100),
                (1, 0, self.placeholder_t_r, -90, 100),
                (2, 0, self.placeholder_t_r, 100, 100),
                (3, 0, self.placeholder_t_r, -110, 100)],
            9: [
                (0, 0, self.placeholder_t_r, 90, 100),
                (1, 0, self.placeholder_t_r, -100, 100),
                (2, 0, self.placeholder_t_r, 110, 100),
                (3, 0, self.placeholder_t_r, -120, 100)],}
        
    def test_calculate_only_neighbors_same_layer(self):
        # Example test, replace with your own cases
        self.initialize()
        self.dic_posToKey_by_batch = {}

        result = calculateOnlyNeighbors(self.dic_int_vars, self.dic_posToKey_by_batch, self.dic_RPhiKey_by_batch, self.dic_list_mcIDs_one_batch, self.dic_all_occupancies, self.batchVars, maxLayer=112)
        
        self.assertEqual(result[0]['NeighborsRemained'], 0)
        
        self.initialize()
        pos1 = (0, 0, 0)
        self.dic_posToKey_by_batch[pos1] = {} #just need to define the keys to indicate a hit there
        # for key in self.dic_posToKey_by_batch_keys:
        #     self.dic_posToKey_by_batch[pos1][key] = np.array([])
        # self.dic_posToKey_by_batch[pos1] = {}
        self.dic_posToKey_by_batch[pos1]['energy_dep_per_cell_RPhiZ'] = [0]
        
        result = calculateOnlyNeighbors(self.dic_int_vars, self.dic_posToKey_by_batch, self.dic_RPhiKey_by_batch, self.dic_list_mcIDs_one_batch, self.dic_all_occupancies, self.batchVars, maxLayer=4)
        
        self.assertEqual(result[0]['NeighborsRemained'], 0)
        
        
        #same layer 2 total
        self.initialize()
        pos2 = (1, 0, 0)
        self.dic_posToKey_by_batch[pos2] = {} 
        self.dic_posToKey_by_batch[pos2]['energy_dep_per_cell_RPhiZ'] = [0]
        self.batchVars['pos_only_neighbors'] = []
        
        result = calculateOnlyNeighbors(self.dic_int_vars, self.dic_posToKey_by_batch, self.dic_RPhiKey_by_batch, self.dic_list_mcIDs_one_batch, self.dic_all_occupancies, self.batchVars, maxLayer=4)
        
        # print(result[5]['pos_only_neighbors'])
        # print(result[0]['NeighborsRemained'])
        
        self.assertEqual(result[0]['NeighborsRemained'], 2) 
        self.assertEqual(result[5]['pos_only_neighbors'], [(0,0,0), (1,0,0)]) #should have both neighbors since same layer
        
        #same layer; 2 has neighbors since outside range
        self.initialize()
        pos2 = (3, 0, 0)
        self.dic_posToKey_by_batch[pos2] = {} 
        # for key in self.dic_posToKey_by_batch_keys:
        #     self.dic_posToKey_by_batch[pos2][key] = np.array([])
        self.dic_posToKey_by_batch[pos2]['energy_dep_per_cell_RPhiZ'] = [0]
        
        result = calculateOnlyNeighbors(self.dic_int_vars, self.dic_posToKey_by_batch, self.dic_RPhiKey_by_batch, self.dic_list_mcIDs_one_batch, self.dic_all_occupancies, self.batchVars, maxLayer=4)
        
        # print(result[5]['pos_only_neighbors'])
        # print(result[0]['NeighborsRemained'])
        
        self.assertEqual(result[0]['NeighborsRemained'], 2) 
        self.assertEqual(result[5]['pos_only_neighbors'], [(0,0,0), (1,0,0)]) #should have both neighbors since same layer
        
        #same layer; 4 has neighbors since outside range
        self.initialize()
        pos3 = (2, 0, 0)
        self.dic_posToKey_by_batch[pos3] = {} 
        # for key in self.dic_posToKey_by_batch_keys:
        #     self.dic_posToKey_by_batch[pos3][key] = np.array([])
        self.dic_posToKey_by_batch[pos3]['energy_dep_per_cell_RPhiZ'] = [0]
        
        # print(f"keys pos: {self.dic_posToKey_by_batch.keys()}")
        # input("Press Enter to continue...")
        
        result = calculateOnlyNeighbors(self.dic_int_vars, self.dic_posToKey_by_batch, self.dic_RPhiKey_by_batch, self.dic_list_mcIDs_one_batch, self.dic_all_occupancies, self.batchVars, maxLayer=4)
        
        # print(result[5]['pos_only_neighbors'])
        # print(result[0]['NeighborsRemained'])
        
        self.assertEqual(result[0]['NeighborsRemained'], 4) 
        self.assertEqual(result[5]['pos_only_neighbors'], [(0,0,0), (1,0,0), (3,0,0), (2,0,0)]) #should have both neighbors since same layer
        
        #out in z
        self.initialize()
        pos5 = (0, 0, 1)
        self.dic_posToKey_by_batch[pos5] = {} 
        # for key in self.dic_posToKey_by_batch_keys:
        #     self.dic_posToKey_by_batch[pos3][key] = np.array([])
        self.dic_posToKey_by_batch[pos5]['energy_dep_per_cell_RPhiZ'] = [0]
        
        # print(f"keys pos: {self.dic_posToKey_by_batch.keys()}")
        # input("Press Enter to continue...")
        
        result = calculateOnlyNeighbors(self.dic_int_vars, self.dic_posToKey_by_batch, self.dic_RPhiKey_by_batch, self.dic_list_mcIDs_one_batch, self.dic_all_occupancies, self.batchVars, maxLayer=4)
        
        # print(result[5]['pos_only_neighbors'])
        # print(result[0]['NeighborsRemained'])
        
        self.assertEqual(result[0]['NeighborsRemained'], 5) 
        self.assertEqual(result[5]['pos_only_neighbors'], [(0,0,0), (1,0,0), (3,0,0), (2,0,0), (0,0,1)]) #should have both neighbors since same layer

    def test_calculate_only_neighbors_across_z(self):
        # Example test, replace with your own cases
        self.initialize()
        self.dic_posToKey_by_batch = {}

        result = calculateOnlyNeighbors(self.dic_int_vars, self.dic_posToKey_by_batch, self.dic_RPhiKey_by_batch, self.dic_list_mcIDs_one_batch, self.dic_all_occupancies, self.batchVars, maxLayer=112)
        
        self.assertEqual(result[0]['NeighborsRemained'], 0)
        
        self.initialize()
        pos1 = (0, 0, 0)
        self.dic_posToKey_by_batch[pos1] = {} #just need to define the keys to indicate a hit there
        self.dic_posToKey_by_batch[pos1]['energy_dep_per_cell_RPhiZ'] = [0]
        
        result = calculateOnlyNeighbors(self.dic_int_vars, self.dic_posToKey_by_batch, self.dic_RPhiKey_by_batch, self.dic_list_mcIDs_one_batch, self.dic_all_occupancies, self.batchVars, maxLayer=4)
        
        self.assertEqual(result[0]['NeighborsRemained'], 0)
        
        
        #same layer 2 total
        self.initialize()
        pos2 = (1, 0, 0)
        self.dic_posToKey_by_batch[pos2] = {} 
        self.dic_posToKey_by_batch[pos2]['energy_dep_per_cell_RPhiZ'] = [0]
        self.batchVars['pos_only_neighbors'] = []
        
        result = calculateOnlyNeighbors(self.dic_int_vars, self.dic_posToKey_by_batch, self.dic_RPhiKey_by_batch, self.dic_list_mcIDs_one_batch, self.dic_all_occupancies, self.batchVars, maxLayer=4)
        
        # print(result[5]['pos_only_neighbors'])
        # print(result[0]['NeighborsRemained'])
        
        self.assertEqual(result[0]['NeighborsRemained'], 2) 
        self.assertEqual(result[5]['pos_only_neighbors'], [(0,0,0), (1,0,0)]) #should have both neighbors since same layer
        
    def test_calculate_only_neighbors_larger_range(self):
        # Example test, replace with your own cases
        self.initialize()
        self.dic_posToKey_by_batch = {}

        pos1 = (0, 0, 0)
        self.dic_posToKey_by_batch[pos1] = {} #just need to define the keys to indicate a hit there
        self.dic_posToKey_by_batch[pos1]['energy_dep_per_cell_RPhiZ'] = [0]
        
        result = calculateOnlyNeighbors(self.dic_int_vars, self.dic_posToKey_by_batch, self.dic_RPhiKey_by_batch, self.dic_list_mcIDs_one_batch, self.dic_all_occupancies, self.batchVars, maxLayer=4)
        
        self.assertEqual(result[0]['NeighborsRemained'], 0)
        
        
        #same layer 2 total
        self.initialize()
        self.dic_int_vars['radiusR'] = 3
        self.dic_int_vars['radiusPhi'] = 3
        self.dic_int_vars['zrange'] = 3
        pos2 = (3, 0, 0)
        self.dic_posToKey_by_batch[pos2] = {} 
        self.dic_posToKey_by_batch[pos2]['energy_dep_per_cell_RPhiZ'] = [0]
        self.batchVars['pos_only_neighbors'] = []
        
        result = calculateOnlyNeighbors(self.dic_int_vars, self.dic_posToKey_by_batch, self.dic_RPhiKey_by_batch, self.dic_list_mcIDs_one_batch, self.dic_all_occupancies, self.batchVars, maxLayer=4)
        
        # print(f"result pos neighbors: {result[5]['pos_only_neighbors']}")
        # print(f"dicpos keys: {self.dic_posToKey_by_batch.keys()}")
        # print(result[0]['NeighborsRemained'])
        
        self.assertEqual(result[0]['NeighborsRemained'], 2) 
        self.assertEqual(result[5]['pos_only_neighbors'], [(0,0,0), (3,0,0)]) #should have both neighbors since same layer
        
        
        #same layer 3 total
        self.initialize()
        self.dic_int_vars['radiusR'] = 3
        self.dic_int_vars['radiusPhi'] = 3
        self.dic_int_vars['zrange'] = 3
        pos2 = (0, 0, 3)
        self.dic_posToKey_by_batch[pos2] = {} 
        self.dic_posToKey_by_batch[pos2]['energy_dep_per_cell_RPhiZ'] = [0]
        self.batchVars['pos_only_neighbors'] = []
        
        result = calculateOnlyNeighbors(self.dic_int_vars, self.dic_posToKey_by_batch, self.dic_RPhiKey_by_batch, self.dic_list_mcIDs_one_batch, self.dic_all_occupancies, self.batchVars, maxLayer=4)
        
        # print(f"result pos neighbors: {result[5]['pos_only_neighbors']}")
        # print(f"dicpos keys: {self.dic_posToKey_by_batch.keys()}")
        # print(result[0]['NeighborsRemained'])
        
        self.assertEqual(result[0]['NeighborsRemained'], 3) 
        self.assertEqual(result[5]['pos_only_neighbors'], [(0,0,0), (3,0,0), (0, 0, 3)]) #should have both neighbors since same layer
        
    def test_calculate_only_neighbors_with_shift(self):
        # Example test, replace with your own cases
        self.initialize()
        self.dic_posToKey_by_batch = {}
        
        #3: [(0, 0, self.placeholder_t_r, 30, 100),
            # (1, 0, self.placeholder_t_r, -40, 100),

        result = calculateOnlyNeighbors(self.dic_int_vars, self.dic_posToKey_by_batch, self.dic_RPhiKey_by_batch, self.dic_list_mcIDs_one_batch, self.dic_all_occupancies, self.batchVars, maxLayer=112)
        
        self.assertEqual(result[0]['NeighborsRemained'], 0)
        
        self.initialize()
        pos1 = (0, 0, 3)
        self.dic_posToKey_by_batch[pos1] = {} #just need to define the keys to indicate a hit there
        self.dic_posToKey_by_batch[pos1]['energy_dep_per_cell_RPhiZ'] = [0]
        
        result = calculateOnlyNeighbors(self.dic_int_vars, self.dic_posToKey_by_batch, self.dic_RPhiKey_by_batch, self.dic_list_mcIDs_one_batch, self.dic_all_occupancies, self.batchVars, maxLayer=4)
        
        self.assertEqual(result[0]['NeighborsRemained'], 0)
        
        
        #same layer 2 total
        self.initialize()
        pos2 = (1, 0, 3)
        self.dic_posToKey_by_batch[pos2] = {} 
        self.dic_posToKey_by_batch[pos2]['energy_dep_per_cell_RPhiZ'] = [0]
        self.batchVars['pos_only_neighbors'] = []
        
        result = calculateOnlyNeighbors(self.dic_int_vars, self.dic_posToKey_by_batch, self.dic_RPhiKey_by_batch, self.dic_list_mcIDs_one_batch, self.dic_all_occupancies, self.batchVars, maxLayer=4, verbose=True)
        
        # print(result[5]['pos_only_neighbors'])
        # print(result[0]['NeighborsRemained'])
        
        self.assertEqual(result[0]['NeighborsRemained'], 2) 
        self.assertEqual(result[5]['pos_only_neighbors'], [(0,0,3), (1,0,3)]) #should have both neighbors since same layer



if __name__ == "__main__":
    unittest.main()
