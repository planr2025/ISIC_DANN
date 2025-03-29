from typing import Optional
import os
from .imagelist import ImageList

class Office31(ImageList):
    """Office31 Dataset.
    
    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a \
            transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    image_list = {
        "amazon": "amazon",
        "dslr": "dslr",
        "webcam": "webcam"
    }
    
    CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
               'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
               'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
               'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

    def __init__(self, root: str, task: str, **kwargs):
        if task not in self.image_list:
            raise ValueError(f"Invalid task '{task}'. Choose from {list(self.image_list.keys())}")
        if "download" in kwargs:
            del kwargs["download"]
        assert task in self.image_list  # Ensure task exists

        # Correct path to the data_list_file inside 'image_list' subfolder
        data_list_file = os.path.join(root, "image_list", self.image_list[task])

        # Check if the data_list_file exists
        if not os.path.exists(data_list_file):
            raise FileNotFoundError(f"Missing file: {data_list_file}. Ensure it's inside {root}/image_list/.")

        super(Office31, self).__init__(root, Office31.CLASSES, data_list_file=data_list_file, **kwargs)


    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
