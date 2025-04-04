import os 
import re 
from typing import List 

def clean_captions() -> List[List[str]]: 
    """
    Returns list of [img_name, cleaned_caption].
    """
    with open('flickr8k/captions.txt', 'r') as file: 
        captions = file.read()
        caption_list = captions.split("\n")
        cleaned_data = []

        assert type(caption_list) is list, "Caption file format is wrong"
        
        for cap in caption_list:
            if cap != "": 
                cap_list = cap.split(",", 1) # first instance of ","
                cleaned =  re.sub(r'[".]+', '', cap_list[1].lower())
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                cap_list[1] = cleaned 

                cleaned_data.append(cap_list)

        return cleaned_data  

def split_dataset(data: List[List[str]], split_ratio: float = 0.8) -> List[List[str]]: 
    """
    Performs splitting of the dataset into train and test. 

    Since each image has multiple captions, the split is performed such that an image only appears in one of the partition. 
    """
    all_images = set(item[0] for item in data)
    all_images = list(all_images)

    import random 
    random.shuffle(all_images) # shuffle the mix to prevent cases of pattern remembering 

    split_idx = int(len(all_images) * split_ratio)
    
    train_split = all_images[:split_idx]
    test_split = all_images[split_idx:]

    train_data = [record for record in data if record[0] in train_split]
    test_data = [record for record in data if record[0] in test_split]

    return train_data, test_data