from scipy import ndimage

def apply_morphological_operations(mask, iterations=2):
    """
    Apply morphological operations to clean up the mask.
    
    Args:
        mask: Binary mask
        iterations: Number of iterations for morphological operations
        
    Returns:
        cleaned_mask: Cleaned binary mask
    """
    # Apply closing to fill small holes
    cleaned_mask = ndimage.binary_closing(mask, iterations=iterations)
    
    # Apply opening to remove small isolated regions
    cleaned_mask = ndimage.binary_opening(cleaned_mask, iterations=iterations)
    
    return cleaned_mask