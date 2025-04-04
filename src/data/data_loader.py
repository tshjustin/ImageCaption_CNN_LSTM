import os 
from PIL import Image 
from typing import List 
from torch.utils.data.dataset import Dataset

"""
The train and test split would have their own Dataloader object
"""
class Flickr8(Dataset): 
    def __init__(self, image_caption_list: List[List[str]], transform=None) -> None: 
        self.image_paths = []
        self.captions = [] 

        for image_caption in image_caption_list: 
            self.image_paths.append(os.path.join("flickr8k/Images", image_caption[0]))
            self.captions.append(image_caption[1])

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.captions[idx]

        image = Image.open(img_path).convert('RGB').resize((224, 224)) # this is the input dimension needed by ResNet-50
        
        if self.transform: 
            image = self.transform(image)
             
        assert image.size == (224, 224), "Image size is not the right dimension needed by ResNet50"

        return image, caption 