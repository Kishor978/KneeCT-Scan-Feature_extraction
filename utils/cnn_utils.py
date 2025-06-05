import numpy as np
import torch


 # Normalize intensity values to [0, 1] range for CNN input
def normalize_volume(volume):
    volume = volume.astype(np.float32)
    # Clip extreme values and normalize
    volume = np.clip(volume, -1000, 3000)  # Typical CT range
    volume = (volume + 1000) / 4000  # Normalize to [0, 1]
    return volume

# Resize volumes to target size for CNN input
def resize_volume(volume, target_size):
    from scipy.ndimage import zoom
    
    current_size = volume.shape
    zoom_factors = [target_size[i] / current_size[i] for i in range(3)]
    resized_volume = zoom(volume, zoom_factors, order=1)  # Linear interpolation
    return resized_volume

def prepare_regions_for_cnn(original_data, femur_mask, tibia_mask, target_size=(64, 64, 64)):
    """
    Prepare the three regions (tibia, femur, background) for CNN input.
    
    Args:
        original_data: Original CT scan data
        femur_mask: Binary mask of femur
        tibia_mask: Binary mask of tibia
        target_size: Target size for resizing volumes (depth, height, width)
        
    Returns:
        regions: Dictionary containing the three regions as tensors
    """
    print("Preparing regions for CNN input...")
    
    # Create background mask (everything that's not bone)
    bone_mask = femur_mask | tibia_mask
    background_mask = ~bone_mask
    
    # Extract regions by masking original data
    tibia_region = original_data.copy()
    tibia_region[~tibia_mask] = 0  # Set non-tibia voxels to 0
    
    femur_region = original_data.copy()
    femur_region[~femur_mask] = 0  # Set non-femur voxels to 0
    
    background_region = original_data.copy()
    background_region[~background_mask] = 0  # Set bone voxels to 0
    
    
    tibia_region = normalize_volume(tibia_region)
    femur_region = normalize_volume(femur_region)
    background_region = normalize_volume(background_region)
        
    if target_size is not None:
        print(f"Resizing volumes from {original_data.shape} to {target_size}")
        tibia_region = resize_volume(tibia_region, target_size)
        femur_region = resize_volume(femur_region, target_size)
        background_region = resize_volume(background_region, target_size)
    
    # Convert to PyTorch tensors and add batch and channel dimensions
    # Shape: (batch_size=1, channels=1, depth, height, width)
    regions = {
        'tibia': torch.from_numpy(tibia_region).unsqueeze(0).unsqueeze(0).float(),
        'femur': torch.from_numpy(femur_region).unsqueeze(0).unsqueeze(0).float(),
        'background': torch.from_numpy(background_region).unsqueeze(0).unsqueeze(0).float()
    }
    
    print(f"Region shapes: Tibia: {regions['tibia'].shape}, Femur: {regions['femur'].shape}, Background: {regions['background'].shape}")
    
    return regions
