import os
import torch
import torchvision.transforms as transforms
from PIL import Image

from data_process import clean_captions, split_dataset
from data_loader import create_vocabulary, create_data_loaders


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def test_dataloaders():
    print("Loading and cleaning captions...")
    caption_data = clean_captions()
    print(f"Total caption entries: {len(caption_data)}")
    
    # Display a few samples
    print("\nSample caption entries:")
    for i in range(3):
        print(f"Image: {caption_data[i][0]}, Caption: {caption_data[i][1]}")
    
    print("\nSplitting data into train and test sets...")
    train_data, test_data = split_dataset(caption_data, split_ratio=0.8)
    print(f"Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # Create vocabulary
    print("\nCreating vocabulary...")
    vocab = create_vocabulary(caption_data, min_word_freq=5)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    batch_size = 4  # Small batch size for testing
    train_loader, test_loader = create_data_loaders(
        train_data, test_data, vocab, transform, batch_size=batch_size
    )
    
    # Test batch from train loader
    print("\nTesting train loader...")
    test_loader_batch(train_loader, vocab, "Train")
    
    # Test batch from test loader
    print("\nTesting test loader...")
    test_loader_batch(test_loader, vocab, "Test")

def test_loader_batch(loader, vocab, loader_name):
    dataiter = iter(loader)
    try:
        images, captions, targets, lengths = next(dataiter)
        
        print(f"\n{loader_name} Loader Batch Information:")
        print(f"Batch size: {images.size(0)}")
        print(f"Image tensor shape: {images.shape}")
        print(f"Caption tensor shape: {captions.shape}")
        print(f"Targets tensor shape: {targets.shape}")
        print(f"Lengths tensor shape: {lengths.shape}")
        
        print("\nSample from batch:")
        for i in range(min(2, images.size(0))):
            print(f"\nSample {i+1}:")
            print(f"Image shape: {images[i].shape}")
            
            # Convert caption indices back to words
            caption_words = [vocab.idx2word[idx.item()] for idx in captions[i][:lengths[i]]]
            caption_str = " ".join(caption_words)
            
            # Convert target index to word
            target_word = vocab.idx2word[targets[i].item()]
            
            print(f"Caption sequence: {caption_str}")
            print(f"Target word: {target_word}")
            print(f"Sequence length: {lengths[i]}")
            
    except StopIteration:
        print(f"No batches available in the {loader_name} loader.")

if __name__ == "__main__":
    test_dataloaders()