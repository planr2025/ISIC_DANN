import os
import pandas as pd
import tempfile
from typing import Optional
from .imagelist import ImageList
from torchvision import transforms
import torch 
class ISIC(ImageList):
    """Dataset Loader for ISIC Dataset (Using CSV for Class Labels).
    
    Args:
        root (str): Root directory of dataset.
        task (str): The domain to create dataset from (e.g., 'domain1', 'domain2').
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(self, root: str, task: str, **kwargs):
        domain_path = os.path.join(root, task)

        if not os.path.isdir(domain_path):
            raise ValueError(f"Invalid domain '{task}'. Folder {domain_path} does not exist.")
        
         # Define strong augmentation
        self.strong_augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
        ])

        csv_file = os.path.join("/u/student/2021/cs21resch15002/DomainShift/DatasetLoaders/isic_download/metadata", f"{task}.csv")
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Missing class mapping CSV file: {csv_file}")
        
        # Load class mappings from CSV
        df = pd.read_csv(csv_file)
        if 'image_id' not in df.columns or 'class' not in df.columns:
            raise ValueError("CSV file must contain 'image_id' and 'class' columns.")

        class_set = set(df['class'].unique())  # Get all unique classes
        class_list = sorted(class_set)  # Ensure a fixed order of class labels
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_list)}  # Map classes to indices

        # Create a temporary file for image list (since ImageList expects a file)
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w')


        with open(temp_file.name, 'w') as f:
            for _, row in df.iterrows():
                img_name = f"{row['image_id']}.jpg"  # Assuming images are stored with .jpg extension
                img_path = os.path.join(domain_path, img_name)
                if os.path.exists(img_path):
                    label = self.class_to_idx[row['class']]
                    f.write(f"{img_path} {label}\n")
                else:
                    print(f"Warning: Image {img_path} not found. Skipping.")

        # Remove 'download' argument if it's in kwargs
        kwargs.pop('download', None)

        # Pass the temp file to ImageList instead of data_list
        super(ISIC, self).__init__(root, class_list, data_list_file=temp_file.name, **kwargs)

        # Remove temp file after ImageList reads it
        os.remove(temp_file.name)


    @classmethod
    def domains(cls, root: str):
        """Lists all available domains (subfolders) in the dataset root."""
        return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

# from PIL import Image
# import numpy as np
# class ISIC(ImageList):
#     """Dataset Loader for ISIC Dataset (Using CSV for Class Labels).
    
#     Args:
#         root (str): Root directory of dataset.
#         task (str): The domain to create dataset from (e.g., 'domain1', 'domain2').
#         transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
#         target_transform (callable, optional): A function/transform that takes in the target and transforms it.
#     """

#     def __init__(self, root: str, task: str, **kwargs):
#         domain_path = os.path.join(root, task)

#         if not os.path.isdir(domain_path):
#             raise ValueError(f"Invalid domain '{task}'. Folder {domain_path} does not exist.")
        
#         # Define strong augmentation
#         self.strong_augment = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomVerticalFlip(p=0.5),
#             transforms.RandomRotation(degrees=30),
#             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#             transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
#             transforms.ToTensor(),
#         ])

#         csv_file = os.path.join("/u/student/2021/cs21resch15002/DomainShift/DatasetLoaders/isic_download/metadata", f"{task}.csv")
#         if not os.path.exists(csv_file):
#             raise FileNotFoundError(f"Missing class mapping CSV file: {csv_file}")
        
#         # Load class mappings from CSV
#         df = pd.read_csv(csv_file)
#         if 'image_id' not in df.columns or 'class' not in df.columns:
#             raise ValueError("CSV file must contain 'image_id' and 'class' columns.")

#         class_set = set(df['class'].unique())  # Get all unique classes
#         class_list = sorted(class_set)  # Ensure a fixed order of class labels
#         self.class_to_idx = {cls: idx for idx, cls in enumerate(class_list)}  # Map classes to indices

#         # Calculate true class distribution
#         class_counts = df['class'].value_counts(normalize=True)
#         class_ratios = {cls: class_counts[cls] for cls in class_list}
#         print(f"Class distribution: {class_ratios}")  # Debug output

#         # Initialize candidate set with true class distribution
#         self.candidate_set = {}
#         for idx, row in df.iterrows():
#             # Get the true class probabilities
#             prob_vector = torch.zeros(len(class_list))
#             for cls, prob in class_ratios.items():
#                 prob_vector[self.class_to_idx[cls]] = prob
            
#             # Normalize to ensure sum=1 (handles floating point errors)
#             prob_vector = prob_vector / prob_vector.sum()
#             self.candidate_set[idx] = prob_vector

#         # Create a temporary file for image list
#         temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
#         with open(temp_file.name, 'w') as f:
#             for idx, row in df.iterrows():
#                 img_name = f"{row['image_id']}.jpg"
#                 img_path = os.path.join(domain_path, img_name)
#                 if os.path.exists(img_path):
#                     label = self.class_to_idx[row['class']]
#                     f.write(f"{img_path} {label}\n")
#                 else:
#                     print(f"Warning: Image {img_path} not found. Skipping.")

#         kwargs.pop('download', None)
#         super(ISIC, self).__init__(root, class_list, data_list_file=temp_file.name, **kwargs)
#         os.remove(temp_file.name)
#     #     # Load candidate set for each image
#     def __getitem__(self, index):
#         img, label = super(ISIC, self).__getitem__(index)

#         # print("Img shape: ", img.shape)
#         if isinstance(img, list):
#             img = img[0]  # Take first element if list

        
#         # Apply augmentations 

#         img_strong = self.strong_augment(img)  # Your strong augmentation pipeline
        
#         img_strong = np.array(img_strong)
        
#         # Get candidate set
#         cs = self.candidate_set[index]
        
#         return (img, img_strong), label, index, cs
#     # def __getitem__(self, index):
#     #     img, label = super(ISIC, self).__getitem__(index)
        
#     #     if isinstance(img, list):
#     #         img = img[0]  # Ensure img is a single tensor
        
#     #     return img, label  # Make sure only 2 values are returned

#     @classmethod
#     def domains(cls, root: str):
#         """Lists all available domains (subfolders) in the dataset root."""
#         return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
