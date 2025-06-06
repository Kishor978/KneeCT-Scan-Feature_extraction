import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_ct_data, visualize_slices, visualize_3d_histograms, visualize_mip, create_animation


def main():
    # Path to the CT scan file
    file_path = 'data\\3702_left_knee.nii.gz'  
    
    # Load the CT scan data
    print("Loading CT scan data...")
    data, affine, header = load_ct_data(file_path)
    
    # Print information about the data
    print(f"CT scan data loaded successfully.")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Value range: [{np.min(data)}, {np.max(data)}]")
    
    # Visualize axial slices
    print("Visualizing axial slices...")
    visualize_slices(data, num_slices=5, axis=2)
    
    # Visualize coronal slices
    print("Visualizing coronal slices...")
    visualize_slices(data, num_slices=5, axis=1)
    
    # Visualize sagittal slices
    print("Visualizing sagittal slices...")
    visualize_slices(data, num_slices=5, axis=0)
        
    # Visualize the histogram
    print("Visualizing histogram...")
    visualize_3d_histograms(data)
    
    # Visualize MIPs
    print("Visualizing Maximum Intensity Projections...")
    visualize_mip(data)
    
    # Create an animation of axial slices
    print("Creating animation of axial slices...")
    anim = create_animation(data, axis=2)
    anim.save('ct_scan_animation.gif', writer='imagemagick', fps=10)
    print("Animation saved as 'ct_scan_animation.gif'")
    anim.event_source.stop()  # Stop the animation event source
    
    anim = create_animation(data, axis=1)
    anim.save('ct_scan_animation_coronal.gif', writer='imagemagick', fps=10)
    print("Animation saved as 'ct_scan_animation_coronal.gif'")
    anim.event_source.stop()  # Stop the animation event source
    anim = create_animation(data, axis=0)
    anim.save('ct_scan_animation_sagittal.gif', writer='imagemagick', fps=10)
    print("Animation saved as 'ct_scan_animation_sagittal.gif'")
    anim.event_source.stop()  # Stop the animation event source
    
    plt.show()

if __name__ == "__main__":
    main()