import numpy as np

def scale_array(arr, target_min, target_max, original_min=None, original_max=None):
    """
    Scale values in array from range [original_min, original_max] to [target_min, target_max]
    
    Args:
        arr: Input array/list to scale
        original_min (A): Minimum value of original range
        original_max (B): Maximum value of original range
        target_min (C): Minimum value of target range
        target_max (D): Maximum value of target range
        
    Returns:
        Scaled array with values mapped to new range
    """
    # print("arr", arr)
    original_min = min(arr) if original_min is  None else original_min
    
    original_max = max(arr) if original_max is None else original_max
    
    # Convert to numpy array if not already
    arr = np.array(arr, dtype=float)
    
    # Apply the scaling formula
    scaled = target_min + (arr - original_min) * (target_max - target_min) / (original_max - original_min)
    
    return scaled



def find_max_index_less_than(sorted_arr, target):
    """
    Find the maximum index in sorted_arr where target < sorted_arr[index]
    
    Args:
        sorted_arr: A sorted array (ascending order)
        target: The target number to compare
    
    Returns:
        The maximum index where target < sorted_arr[index]
        Returns -1 if no such index exists
    """
    left, right = 0, len(sorted_arr) - 1
    result = len(sorted_arr)  # Default to the length of the array
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if target < sorted_arr[mid]:
            # This index works, but we might find a smaller one
            result = mid
            right = mid - 1
        else:
            # Need to look in the right half
            left = mid + 1
            
    return result

def find_closest_point_after_redistribution(X, location_A_index, added_points=48):
    """
    Find the closest point to locationA after redistribution with more points.
    
    Args:
        X: Original number of points dividing the circle
        location_A_index: Index of locationA in original division (0 to X-1)
        added_points: Number of points added (default=48)
    
    Returns:
        tuple: (closest_point_index, angular_shift)
            - closest_point_index: Index of closest point in new division
            - angular_shift: Angular shift between original and new point (radians)
    """
    # Calculate angle of locationA in the original division
    angle_A = 2 * np.pi * location_A_index / X
    
    # New points per circle
    new_X = X + added_points
    
    # Calculate theoretical index in new division
    theoretical_index = angle_A * new_X / (2 * np.pi)
    
    # Find closest actual index
    closest_index = round(theoretical_index)
    closest_index = closest_index % new_X  # Ensure it's within range
    
    # Calculate angle of the closest point
    closest_angle = 2 * np.pi * closest_index / new_X
    
    # Calculate angular shift
    relative_angular_shift = closest_angle - angle_A
    # may need to double check if we need to switch the sign
    
    return closest_index, relative_angular_shift

def check_odd_fractions(A, B, max_k=160, offset = None, verbose = False):
    """
    Check if A is greater than fractions of form (k/4)B where k is odd. I.e. used where we check if some phi shift is greater than 1/4 phi step, 3/4 phi step, etc.
    
    Args:
        A: The number to compare
        B: The constant factor
        max_k: Maximum odd k to check (optional, default=10)
        
    Returns:
        tuple (comparison_result, k_value) where comparison_result is True if A is greater than any of the odd fractions, and k_value is the largest odd k for which A >= (k/4)B.
    """
    # # Calculate 4A/B
    # ratio = 4 * A / B
    
    # # Find largest odd integer n where n ≤ 4A/B
    # n = int(ratio)
    # if n % 2 == 0:  # If n is even, take previous odd
    #     n -= 1
        
    # Check each odd fraction up to max_k
    resultN = 0
    resultComparison = False
    signA = np.sign(A)
    if offset is not None:
        A = A + offset
    absA = np.abs(A)
    # for k in range(1, max_k + 1, 2):
    for k in range(1, max_k + 1, 1):
        if k == max_k: #max shift is 183 with step of 37 which is 20 odd fractions, so we can warn the user if we reach this point
            print(f"Warning: max_k {max_k} reached, A may not be greater than all odd fractions. Where A = {A}, signA = {signA}, B = {B}, resultN = {resultN}, resultComparison = {resultComparison}")
            # input("Press Enter to continue, or Ctrl+C to exit.")
        fraction = (k/4) * B
        if verbose:
            print(f"Checking {k}/4B = {fraction} against A = {A}")
        comparison= absA >= fraction
        if comparison: #if A is greater than or equal to the fraction, we set keep going
            resultComparison = True
            resultN = k * signA
        else: #if A is not greater than the fraction, we break
            break
        
    
    # if resultComparison:      
    #     return True, resultN
    # else:
    #     return False, 0
    return resultComparison, resultN
        # results[f"{k}/4B = {fraction}"] = comparison
        
        
def fast_check_odd_fractions(A, B, max_k=160, offset = None, verbose = False):
    """
    Check if A is greater than fractions of form (k/4)B where k is odd. I.e. used where we check if some phi shift is greater than 1/4 phi step, 3/4 phi step, etc.
    
    Args:
        A: The number to compare
        B: The constant factor
        max_k: Maximum odd k to check (optional, default=10)
        
    Returns:
        tuple (comparison_result, k_value) where comparison_result is True if A is greater than any of the odd fractions, and k_value is the largest odd k for which A >= (k/4)B.
    """
    # Calculate 4A/B
    ratio = 4 * A / B
    
    # # Find largest odd integer n where n ≤ 4A/B
    # n = int(ratio)
    # if n % 2 == 0:  # If n is even, take previous odd
    #     n -= 1
        
    # Check each odd fraction up to max_k
    resultN = 0
    resultComparison = False
    signA = np.sign(A)
    if offset is not None:
        A = A + offset
    absA = np.abs(A)

    # Prepare the list of fractions (k/4)*B for k in 1..max_k
    fractions = np.array([(k / 4) * B for k in range(1, max_k + 1)])
    # Binary search for the largest k where absA >= (k/4)*B
    left, right = 0, max_k - 1
    idx = -1
    while left <= right:
        mid = (left + right) // 2
        if absA >= fractions[mid]:
            idx = mid
            left = mid + 1
        else:
            right = mid - 1

    if idx >= 0:
        resultComparison = True
        resultN = (idx + 1) * signA  # k = idx+1 since k starts from 1
    else:
        resultComparison = False
        resultN = 0
    # print(f"fast_check_odd_fractions: A = {A}, signA = {signA}, B = {B}, resultN = {resultN}, resultComparison = {resultComparison}")

    if idx == max_k - 1:
        print(f"Warning: max_k {max_k} reached, A may not be greater than all odd fractions. Where A = {A}, signA = {signA}, B = {B}, resultN = {resultN}, resultComparison = {resultComparison}, and offset = {offset}")
    return resultComparison, resultN
    
    
def faster_check_odd_fractions(A, B, max_k=160, offset = None, verbose = False):
    resultN = int(4 * A // B)
    if resultN == 0:
        resultComparison = False
    else:
        resultComparison = True

    return resultComparison, resultN


def find_closest_indices(arr, x, n=10): # may delete
    """
    Find n indices of values in sorted array closest to x
    
    Parameters:
    arr - sorted array
    x - target value
    n - number of closest values to find (default 10)
    
    Returns:
    List of n indices, sorted by proximity to target value
    """
    # Find index of closest value
    idx = np.searchsorted(arr, x)
    
    # Handle edge cases
    if idx == 0:
        return list(range(min(n, len(arr))))
    if idx == len(arr):
        return list(range(max(len(arr) - n, 0), len(arr)))
    
    # Create window around the closest point and expand outward
    left = idx - 1
    right = idx
    indices = []
    
    # Collect n closest points
    while len(indices) < n and (left >= 0 or right < len(arr)):
        # If we've hit array bounds, only consider the available side
        if left < 0:
            indices.append(right)
            right += 1
        elif right >= len(arr):
            indices.append(left)
            left -= 1
        # Compare distances and pick closest
        elif abs(arr[left] - x) <= abs(arr[right] - x):
            indices.append(left)
            left -= 1
        else:
            indices.append(right)
            right += 1
    
    # Sort indices by proximity to target value
    return sorted(indices, key=lambda i: abs(arr[i] - x))

def kappa(L, phi_d):
    """Defines the kappa of the wire, which is a constant based on the length and twist angle

    Args:
        L (float): Total length of the wire (in mm) (defined to be the total length of detector)
        phi_d (float): Twist angle of the wire (in radians)

    Returns:
        float: kappa constant
    """
    k = 2*L / np.tan(phi_d/2)
    return k



def globalPhiIndex(zpos, z_layer_to_shift, n_cell_per_layer): #may delete
    """Given the z position and the z_layer_to_shift, this function returns the global phi index for each layer.

    Args:
        zpos (list): list of z positions
        z_layer_to_shift (dict): dictionary which maps z position to a list of tuples (layer, t_r, theta) for each layer
        n_cell_per_layer (int): number of cells per layer

    Returns:
        dict: dictionary which maps z position to a list of tuples (layer, t_r, theta) for each layer
    """
    
    global_phi_index = {}
    for i, z in enumerate(zpos):
        global_phi_index[z] = []
        for j in range(0, len(z_layer_to_shift[z])):
            total_n_cells_in_layer = n_cell_per_layer[j]
            phi_step = 2 * np.pi / total_n_cells_in_layer
            # global_phi = 
            
            global_phi_index[z].append((j, z_layer_to_shift[z][j][1], z_layer_to_shift[z][j][2]))
            # layer, global r,  
    return global_phi_index