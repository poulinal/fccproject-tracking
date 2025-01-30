import numpy as np
import matplotlib.pyplot as plt

# Cylinder and Layer Definitions
num_layers = 112  # Total layers
# layer_radii = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])  # Radii of each layer
layer_radii = np.arange(1, num_layers + 1, 1)  # Radii of each layer
num_cells_per_layer = np.array([8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34])  # Cells per layer

# print(len(layer_radii))

imageOutputPath = "fccproject-tracking/detector_beam_backgrounds/tracking/images/cylinder_layers.png"


def plot_wire_chamber(num_layers, num_cells_per_layer, imageOutputPath=imageOutputPath, layer_radii=layer_radii, title="Cylinder Layers with Scattered Cells"):
    # print(num_cells_per_layer)
    n_cells_per_layer = num_cells_per_layer.values()
    layer_radii *= 20  # Double radii for better visualization
    layer_radii += 349
    # print(n_cells_per_layer)
    #check right size of num_cells_per_layer
    if len(num_cells_per_layer) != num_layers:
        raise ValueError("num_cells_per_layer must have the same length as num_layers")
    if len(layer_radii) != num_layers:
        raise ValueError("layer_radii must have the same length as num_layers")
    
    # Create Polar Plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Generate Cells in Each Layer
    for radius, num_cells in zip(layer_radii, n_cells_per_layer):
        theta = np.linspace(0, 2 * np.pi, num_cells, endpoint=False)  # Evenly distribute cells
        r = np.full_like(theta, radius)  # Set radius for all points in this layer
        
        ax.scatter(theta, r, s=2, marker='.', label=f'Layer {radius}', alpha=0.7)  # Scatter plot for cells

    #draw a line circle at 349
    ax.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, 349), color='black', linestyle='--', linewidth=1, label='Vertex')    
    
    # Styling
    ax.set_title(title)
    ax.set_yticklabels([])  # Hide radial labels for cleaner look
    # ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))  # Move legend to the side

    #save
    plt.savefig(imageOutputPath, dpi=300, bbox_inches='tight')


