import numpy as np
import matplotlib.pyplot as plt

def visualize_slices(data, num_slices=5, axis=2):
    """
    Visualize multiple slices of the CT scan along a specified axis.
    
    Args:
        data: CT scan data as a numpy array
        num_slices: Number of slices to visualize
        axis: Axis along which to take slices (0=sagittal, 1=coronal, 2=axial)
    """
    # Get the size of the data along the specified axis
    axis_size = data.shape[axis]
    
    # Calculate the indices of the slices to visualize
    indices = np.linspace(0, axis_size - 1, num_slices, dtype=int)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 3))
    
    # Axis labels
    axis_names = ['Sagittal', 'Coronal', 'Axial']
    
    # Loop through the indices and visualize each slice
    for i, idx in enumerate(indices):
        # Take a slice along the specified axis
        if axis == 0:
            slice_data = data[idx, :, :]
        elif axis == 1:
            slice_data = data[:, idx, :]
        else:  # axis == 2
            slice_data = data[:, :, idx]
        
        # Display the slice
        axes[i].imshow(slice_data.T, cmap='bone', origin='lower')
        axes[i].set_title(f'{axis_names[axis]} Slice {idx}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_3d_histograms(data, bins=50):
    """
    Visualize the histogram of voxel intensities in the CT scan.
    
    Args:
        data: CT scan data as a numpy array
        bins: Number of bins for the histogram
    """
    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Plot the histogram
    plt.hist(data.flatten(), bins=bins)
    plt.title('Histogram of CT Scan Voxel Intensities')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.yscale('log')  # Use log scale for better visualization
    
    # Add a vertical line at a potential threshold for bone segmentation
    # This is a starting point that will need to be adjusted
    threshold = np.percentile(data, 95)  # 95th percentile as an initial guess
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Potential Threshold ({threshold:.2f})')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def create_mip_projection(data, axis=2):
    """
    Create a Maximum Intensity Projection (MIP) along a specified axis.
    
    Args:
        data: CT scan data as a numpy array
        axis: Axis along which to project (0=sagittal, 1=coronal, 2=axial)
        
    Returns:
        mip: Maximum Intensity Projection as a 2D numpy array
    """
    # Create the MIP by taking the maximum value along the specified axis
    mip = np.max(data, axis=axis)
    
    return mip

def visualize_mip(data):
    """
    Visualize Maximum Intensity Projections (MIPs) along all three axes.
    
    Args:
        data: CT scan data as a numpy array
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axis names
    axis_names = ['Sagittal', 'Coronal', 'Axial']
    
    # Loop through the axes and create MIPs
    for i in range(3):
        # Create the MIP
        mip = create_mip_projection(data, axis=i)
        
        # Display the MIP
        axes[i].imshow(mip.T, cmap='bone', origin='lower')
        axes[i].set_title(f'{axis_names[i]} MIP')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def visualize_segmentation(original_data, femur_mask, tibia_mask, num_slices=5, axis=2, save_path=None):
    """
    Visualize the segmentation results with overlays for femur and tibia.
    
    Args:
        original_data: Original CT scan data (numpy array)
        femur_mask: Binary mask of the femur
        tibia_mask: Binary mask of the tibia
        num_slices: Number of slices to visualize
        axis: Axis along which to slice (0=sagittal, 1=coronal, 2=axial)
        save_path: If provided, path to save the figure
    """
    axis_size = original_data.shape[axis]
    margin = min(50, axis_size // 6)

    if axis_size < 2 * margin:
        raise ValueError(f"Volume too small along axis {axis} to apply margin safely.")

    indices = np.linspace(margin, axis_size - margin - 1, num_slices, dtype=int)

    fig, axes = plt.subplots(3, num_slices, figsize=(15, 9))
    axis_names = ['Sagittal', 'Coronal', 'Axial']

    for i, idx in enumerate(indices):
        if axis == 0:
            original_slice = original_data[idx, :, :]
            femur_slice = femur_mask[idx, :, :]
            tibia_slice = tibia_mask[idx, :, :]
        elif axis == 1:
            original_slice = original_data[:, idx, :]
            femur_slice = femur_mask[:, idx, :]
            tibia_slice = tibia_mask[:, idx, :]
        else:  # axis == 2
            original_slice = original_data[:, :, idx]
            femur_slice = femur_mask[:, :, idx]
            tibia_slice = tibia_mask[:, :, idx]

        norm_slice = np.clip(original_slice, -300, 1500)
        range_val = np.max(norm_slice) - np.min(norm_slice)
        if range_val > 0:
            norm_slice = (norm_slice - np.min(norm_slice)) / range_val
        else:
            norm_slice = np.zeros_like(norm_slice)

        # Row 1: original
        axes[0, i].imshow(norm_slice, cmap='bone', origin='lower')
        axes[0, i].set_title(f'{axis_names[axis]} Slice {idx} - Original')
        axes[0, i].axis('off')

        # Row 2: femur overlay
        axes[1, i].imshow(norm_slice, cmap='bone', origin='lower')
        femur_overlay = np.zeros_like(norm_slice, dtype=np.float32)
        femur_overlay[femur_slice] = 1.0
        axes[1, i].imshow(femur_overlay, cmap='Reds', alpha=0.5, origin='lower')
        axes[1, i].set_title(f'{axis_names[axis]} Slice {idx} - Femur')
        axes[1, i].axis('off')

        # Row 3: tibia overlay
        axes[2, i].imshow(norm_slice, cmap='bone', origin='lower')
        tibia_overlay = np.zeros_like(norm_slice, dtype=np.float32)
        tibia_overlay[tibia_slice] = 1.0
        axes[2, i].imshow(tibia_overlay, cmap='Blues', alpha=0.5, origin='lower')
        axes[2, i].set_title(f'{axis_names[axis]} Slice {idx} - Tibia')
        axes[2, i].axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
