import os

# Define class names and create a mapping from class name to index
CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
           'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
           'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
           'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']
class_to_index = {cls: i for i, cls in enumerate(CLASSES)}

# Define paths
dataset_root = "/u/student/2021/cs21resch15002/DomainShift/Datasets/Office31/webcam"
output_file = "/u/student/2021/cs21resch15002/DomainShift/Datasets/Office31/image_list/webcam"

# Create webcam.txt
with open(output_file, "w") as f:
    for class_name in os.listdir(dataset_root):  # Loop through class folders
        class_path = os.path.join(dataset_root, class_name)
        if os.path.isdir(class_path) and class_name in class_to_index:
            label = class_to_index[class_name]  # Get class index
            for img in os.listdir(class_path):  # Loop through images
                img_path = os.path.join("webcam", class_name, img)  # Relative path
                f.write(f"{img_path} {label}\n")

print(f"✅ webcam.txt successfully generated at: {output_file}")
