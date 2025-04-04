import os 
import re 
from typing import List 

def clean_captions() -> List[List[str]]: 
    """
    Returns list of [img_name, cleaned_caption.
    """
    with open('flickr8k/captions.txt', 'r') as file: 
        captions = file.read()
        caption_list = captions.split("\n")
        cleaned_captions = []

        assert type(caption_list) is list, "Caption file format is wrong"
        
        for cap in caption_list:
            if cap != "": 
                cap_list = cap.split(",", 1) # first instance of ","
                cleaned =  re.sub(r'[".]+', '', cap_list[1].lower())
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                cap_list[1] = cleaned 

                cleaned_captions.append(cap_list)

        return clean_captions


