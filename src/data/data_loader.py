import os
import torch
from PIL import Image
from typing import List
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


"""
The dataloader would accomplish the following task: For a caption of length n, generate n-1 samples. 

Example:  <start> happy dog <end>
------------------------

<start>
<start> happy
<start> happy dog

"""
# This loader is not needed - It was initially tested for to load the images - captions in batches 
# class Flickr8(Dataset): 
#     def __init__(self, image_caption_list: List[List[str]], transform=None) -> None: 
#         self.image_paths = []
#         self.captions = [] 

#         for image_caption in image_caption_list: 
#             self.image_paths.append(os.path.join("flickr8k/Images", image_caption[0]))
#             self.captions.append(image_caption[1])

#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         caption = self.captions[idx]

#         image = Image.open(img_path).convert('RGB').resize((224, 224)) # this is the input dimension needed by ResNet-50
        
#         if self.transform: 
#             image = self.transform(image)
             
#         assert image.size == (224, 224), "Image size is not the right dimension needed by ResNet50"

#         return image, caption 

class Vocabulary:
    """
    Builds a vocabulary from the dataset
    """
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.idx = 4
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)
    
    def build_vocab(self, captions, threshold=3):
        counter = Counter()
        for caption in captions:
            counter.update(caption.split())
            
        words = [word for word, count in counter.items() if count >= threshold]
        
        for word in words:
            self.add_word(word)

class FlickrPartialSequenceDataset(Dataset):
    def __init__(self, image_caption_list: List[List[str]], vocab, transform=None):
        """
        For a sequence of length n, we want to generate n-1 sequences 

        <start> wet dog <end>

        <start>
        <start> wet 
        <start> wet dog

        image_path = List of image path  -> ['1.jpg', '2.jpg',...]
        captions = [[<start>], [<start>, wet], [<start>, wet, dog]]
        targets = ['wet', 'dog', '<end>']
        
        """
        self.vocab = vocab
        self.transform = transform
        
        self.image_paths = []
        self.captions = []
        self.targets = []
        self.lengths = []
        
        for image_caption in image_caption_list:
            image_path = os.path.join("flickr8k/Images", image_caption[0])
            caption = image_caption[1]
            
            tokens = ["<start>"] + caption.split() + ["<end>"] # eg: ['<start>', 'woman', 'writing', 'on', 'a', 'pad', 'in', 'room', 'with', 'gold', ',', 'decorated', 'walls', '<end>']
            
            for i in range(len(tokens) - 1):
                sequence = tokens[:i+1]
                target = tokens[i+1]
                
                self.image_paths.append(image_path)
                self.captions.append(sequence)
                self.targets.append(target)
                self.lengths.append(len(sequence))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.captions[idx]
        target = self.targets[idx]
        length = self.lengths[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # converts the word to index
        caption_indices = [self.vocab(word) for word in caption]
        
        target_index = self.vocab(target)
        
        return image, torch.tensor(caption_indices), target_index, length
    
def collate_fn(batch): 
    """
    Notice that in the above definition, the order of the sequences is in the same order as the targets, meaning that caption[0]'s target = target[0]. 

    We want to maintain this mapping such that we are using the correct target labels 

    Another issue is that during training of the LSTM, there is an obvious issue of varying length of input sequence. Since we need to run through all the state, 
    we would need to use padding.

    collate_fn essnetially is a glue that allows us to specify the way examples stick together in a batch + add more modifications  


    Since we are throwing in a batch of data, we need to represent them as a "block" of data. To do so: 
    1. For images since same shape: stack (piles up the data in a specified dimension - Requires each entry to be of the same size 
       (recall size = total number of elements | shape = dimensions of object))

    2. For captions, since of difference lengths, 
    """
    batch.sort(key=lambda x: x[3], reverse=True)
    images, captions, targets, lengths = zip(*batch) # unpack - images = [img1, img2,...] | captions = [cap1, cap2... (of varying length)]
    images = torch.stack(images, 0) # stack to single tensor 
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0) # pad all the caption to same length => outputs = (B * T), B=batchsize, T=length of longest seq

    targets = torch.tensor(targets)
    lengths = torch.tensor(lengths)

    return images, captions_padded, targets, lengths


def get_partial_sequence_loader(image_caption_list, vocab, transform, batch_size=32, shuffle=True, num_workers=4):
    dataset = FlickrPartialSequenceDataset(image_caption_list, vocab, transform)
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return data_loader


def create_vocabulary(caption_data, min_word_freq=5):
    vocab = Vocabulary()
    all_captions = [item[1] for item in caption_data]
    vocab.build_vocab(all_captions, threshold=min_word_freq) # builds the vocab from the flick8k  
    return vocab

def create_data_loaders(train_data, test_data, vocab, transform, batch_size=32):
    train_loader = get_partial_sequence_loader(
        train_data, vocab, transform, batch_size=batch_size
    )
    
    test_loader = get_partial_sequence_loader(
        test_data, vocab, transform, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader