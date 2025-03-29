import pandas as pd

def convert_csv_to_txt(csv_file, output_txt):
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Open the output text file
    with open(output_txt, 'w') as f:
        for _, row in df.iterrows():
            image_path = f"/u/student/2021/cs21resch15002/DomainShift/Datasets/ISIC_Dataset/bcn_default/{row['image_id']}.jpg"
            class_label = row['class']
            f.write(f"{image_path} {class_label}\n")
    
    print(f"File saved as {output_txt}")

# Example usage
csv_file = "DatasetLoaders/isic_download/metadata/bcn_default.csv"  # Replace with your actual CSV file
output_txt = "Datasets/ISIC_Dataset/image_list/bcn_default"  # Desired output file name
convert_csv_to_txt(csv_file, output_txt)
