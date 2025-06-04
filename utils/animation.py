from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import numpy as np
import os

def create_animation(data, axis=2):
    """
    Create an animation of slices along a specified axis.
    
    Args:
        data: CT scan data as a numpy array
        axis: Axis along which to animate (0=sagittal, 1=coronal, 2=axial)
        
    Returns:
        anim: Animation object
    """
    # Get the size of the data along the specified axis
    axis_size = data.shape[axis]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Axis names
    axis_names = ['Sagittal', 'Coronal', 'Axial']
    
    # Function to update the plot for each frame
    def update(frame):
        ax.clear()
        
        # Take a slice along the specified axis
        if axis == 0:
            slice_data = data[frame, :, :]
        elif axis == 1:
            slice_data = data[:, frame, :]
        else:  # axis == 2
            slice_data = data[:, :, frame]
        
        # Display the slice
        ax.imshow(slice_data.T, cmap='bone', origin='lower')
        ax.set_title(f'{axis_names[axis]} Slice {frame}')
        ax.axis('off')
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=axis_size, interval=50)
    
    return anim

def create_segmentation_animation(data, femur_mask=None, tibia_mask=None, axis=2, interval=50, save_path=None):
    """
    Create an animation of slices along a specified axis with optional femur and tibia overlays.
    
    Args:
        data: 3D CT scan data as a numpy array
        femur_mask: Binary mask of femur (same shape as data)
        tibia_mask: Binary mask of tibia (same shape as data)
        axis: Axis along which to animate (0=sagittal, 1=coronal, 2=axial)
        interval: Time between frames in milliseconds
        save_path: If provided, saves animation to given path (.gif or .mp4)
        
    Returns:
        anim: Animation object
    """
    axis_size = data.shape[axis]
    fig, ax = plt.subplots(figsize=(8, 8))
    axis_names = ['Sagittal', 'Coronal', 'Axial']

    def update(frame):
        ax.clear()
        
        # Extract slices
        if axis == 0:
            slice_data = data[frame, :, :]
            femur_slice = femur_mask[frame, :, :] if femur_mask is not None else None
            tibia_slice = tibia_mask[frame, :, :] if tibia_mask is not None else None
        elif axis == 1:
            slice_data = data[:, frame, :]
            femur_slice = femur_mask[:, frame, :] if femur_mask is not None else None
            tibia_slice = tibia_mask[:, frame, :] if tibia_mask is not None else None
        else:
            slice_data = data[:, :, frame]
            femur_slice = femur_mask[:, :, frame] if femur_mask is not None else None
            tibia_slice = tibia_mask[:, :, frame] if tibia_mask is not None else None

        # Normalize CT slice
        slice_data = np.clip(slice_data, -300, 1500)
        norm = slice_data - slice_data.min()
        norm = norm / norm.max() if norm.max() > 0 else np.zeros_like(norm)

        ax.imshow(norm.T, cmap='bone', origin='lower')

        if femur_slice is not None:
            ax.imshow(femur_slice.T, cmap='Reds', alpha=0.4, origin='lower')
        if tibia_slice is not None:
            ax.imshow(tibia_slice.T, cmap='Blues', alpha=0.4, origin='lower')

        ax.set_title(f'{axis_names[axis]} Slice {frame}')
        ax.axis('off')

    anim = FuncAnimation(fig, update, frames=axis_size, interval=interval)

    # Save if requested
    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        if ext == '.gif':
            anim.save(save_path, writer='pillow', fps=1000 // interval)
        elif ext == '.mp4':
            anim.save(save_path, writer='ffmpeg', fps=1000 // interval)
        print(f"Animation saved to {save_path}")

    return anim