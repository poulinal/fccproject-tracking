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



def check_odd_fractions(A, B, max_k=10):
    """
    Check if A is greater than fractions of form (k/4)B where k is odd. I.e. used where we check if some phi shift is greater than 1/4 phi step, 3/4 phi step, etc.
    
    Args:
        A: The number to compare
        B: The constant factor
        max_k: Maximum odd k to check (optional, default=10)
        
    Returns:
        Dictionary mapping each fraction to True/False (True if A > fraction)
    """
    # # Calculate 4A/B
    # ratio = 4 * A / B
    
    # # Find largest odd integer n where n ≤ 4A/B
    # n = int(ratio)
    # if n % 2 == 0:  # If n is even, take previous odd
    #     n -= 1
        
    # Check each odd fraction up to max_k
    results = {}
    for k in range(1, max_k + 1, 2):
        if k == max_k:
            print(f"Warning: max_k {max_k} reached, A may not be greater than all odd fractions.")
        fraction = (k/4) * B
        comparison = A > fraction
        if comparison:
            return True, k
        if k == 1 and not comparison:
            return False, -1
        # results[f"{k}/4B = {fraction}"] = comparison


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