import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from skimage.measure import block_reduce
from torch.utils.data import Dataset

# *** Custom class for importing MRI and labels ***
class CustomMRIDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_transform=None, filter_size=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with the folder names and labels.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            target_size (tuple, optional): Target size for downsampling (new_height, new_width, new_depth).
        """
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.filter_size = filter_size
    
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

        if self.filter_size:
            img_np = block_reduce(img_np, self.filter_size, np.mean)
        
        img_np = np.expand_dims(img_np, axis=0)

        # Convert to tensor
        img_tensor = torch.from_numpy(img_np)

        # Get the label
        label = self.img_labels.iloc[idx, 1]
        label_tensor = torch.tensor([label], dtype=torch.float32)

        if self.transform:
            img_tensor = self.transform(img_tensor)
        if self.target_transform:
            label_tensor = self.target_transform(label_tensor)

        return img_tensor, label_tensor