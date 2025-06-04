import nibabel as nib
import os
import numpy as np
def load_ct_data(file_path):
    """
    Load CT scan data from a file.
    
    Args:
        file_path: Path to the CT scan file
        
    Returns:
        data: CT scan data as a numpy array
        affine: Affine transformation matrix
        header: Header information from the file
    """
    # Load the CT scan using nibabel
    ct_img = nib.load(file_path)
    
    # Get the data as a numpy array
    data = ct_img.get_fdata()
    
    # Get the affine transformation matrix
    affine = ct_img.affine
    
    # Get the header information
    header = ct_img.header
    
    return data, affine, header

def save_nifti(data, affine, header, output_path):
    """
    Save data as a NIfTI file.
    
    Args:
        data: Data to save
        affine: Affine transformation matrix
        header: Header information
        output_path: Path to save the file
    """
    # Create a directory for the output if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a NIfTI image
    img = nib.Nifti1Image(data.astype(np.int16), affine, header)
    
    # Save the image
    nib.save(img, output_path)
    
    print(f"Saved: {output_path}")
