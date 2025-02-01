import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from skimage.measure import block_reduce
from torch.utils.data import Dataset

# *** Custom class for importing MRI and labels ***
class CustomMRIDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with the folder names and labels.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        # get the 'subj' and 'ses' values from csv_file
        subj = self.img_labels.iloc[idx, 0]
        ses = self.img_labels.iloc[idx, 2]

        # Construct the image path
        img_path = os.path.join(self.img_dir, subj, ses, 'anat', subj + '_' + ses + '_T2w.nii.gz')

        # Read a NIfTI file
        img_nii = nib.load(img_path)
        img_np = img_nii.get_fdata()
        img_np = img_np.astype(np.float32)

        # Slice the 3D image along the depth dimension
        num_slices = img_np.shape[0]
        slices = []
        pool_size = 10 # For avg/max pooling to reduce the number of slices (GPU memory issue)
        for i in range(0, num_slices, pool_size):
            slice_3d = img_np[i:i+pool_size, :, :] # Get a 3D block
            slice_2d = block_reduce(slice_3d, (pool_size, 1, 1), np.mean) # Apply avg pooling
            slices.append(slice_2d)
        
        # Stack the slices along the batch dimension
        img_np_stacked = np.stack(slices, axis=0)

        # Convert to tensor
        img_tensor = torch.from_numpy(img_np_stacked)

        # Get the label
        label = self.img_labels.iloc[idx, 1]
        label_tensor = torch.tensor([label], dtype=torch.float32)

        if self.transform:
            img_tensor = self.transform(img_tensor)
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)

        return img_tensor, label_tensor