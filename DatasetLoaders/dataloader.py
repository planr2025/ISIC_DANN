import kagglehub
import os
from tqdm import tqdm  # For the progress bar

# Define the target folder where you want to download the dataset
target_folder = "/u/student/2021/cs21resch15002/DomainShift/Datasets"

# Make sure the folder exists, or create it
os.makedirs(target_folder, exist_ok=True)

# Download the dataset to the specified folder with a progress bar
# Assuming kagglehub has a function to provide file size, which tqdm uses
path = kagglehub.dataset_download("xixuhu/office31", path=target_folder)

# Use tqdm if there is an iterable or if we need to monitor the size manually
with tqdm(total=100, desc="Downloading Dataset", unit="%", ncols=100) as pbar:
    pbar.update(100)  # This is a placeholder until the actual download is processed

print("Path to dataset files:", path)
