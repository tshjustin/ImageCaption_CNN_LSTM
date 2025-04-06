import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from model.encode_visuals import EncoderCNN
from model.lstm import LSTMDecoder
from data.glove import integrate_glove_embeddings
from data.data_process import clean_captions, split_dataset
from data.data_loader import create_vocabulary, create_data_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    encoder.train()
    decoder.train()
    
    # total loss of epoch 
    total_loss = 0
    
    for i, (images, captions, targets, lengths) in enumerate(train_loader):
        """
        i = batch_index 

        images.shape = [32, 3, 224, 224]
        captions.shape = [32, 5] => 32 captions, with max length = 5 => [[1,2,3,4,5], [1,2,3,4,5]] ... 
        targets.shape = [32] => 32 GT, wtih length = 1 => [3, 2,....]
        length.shape = [32] => [...]        
        """
        images = images.to(device)
        captions = captions.to(device)
        targets = targets.to(device)
        
        # clear gradient of prv batch 
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        features = encoder(images)
        outputs = decoder(features, captions, lengths) # .shape = ( # sequences, # timesteps, # vocab size). This runs the whole batch under the LSTM 
        
        # takes the total number of sequences => returns the size of a specific position 
        batch_size = outputs.size(0)
        
        # total loss of the batch 
        loss = 0
        
        """
        Example outputs: 
        outputs = 
        [
            # Sequence 1
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5],  # <start> position
                [0.2, 0.1, 0.5, 0.3, 0.4, 0.2, 0.1, 0.5, 0.3, 0.4],  # dog position
            *   [0.3, 0.4, 0.1, 0.5, 0.2, 0.6, 0.1, 0.2, 0.3, 0.1],  # runs position (need to predict after this)
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # padding
            ],

            // at "runs", we would predict for the word that comes after it!. Note that at each cell state, we are predicting for the NEXT word. 
            
            # Sequence 2
            [
                [0.2, 0.3, 0.1, 0.4, 0.5, 0.2, 0.3, 0.1, 0.4, 0.5],  # <start> position
                [0.1, 0.2, 0.3, 0.5, 0.4, 0.1, 0.2, 0.3, 0.5, 0.4],  # cat position
                [0.5, 0.4, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.1],  # sleeps position
                [0.3, 0.2, 0.4, 0.1, 0.5, 0.3, 0.7, 0.4, 0.1, 0.5],  # on position (need to predict after this)
            ],
        ]
        """

        # for each sample in the batch
        for j in range(batch_size):

            # take the actual, not padded length of that sequence. .item() returns the size of that array
            seq_length = lengths[j].item() - 1

            # get the logits of the next word after the last token in sequnce j - Look at the example above to see what we are extracting! 
            # 1. get the jth sequence 
            # 2. get the n-th word - in the example shown with a *
            # 3. copy that array extracted 
            prediction = outputs[j, seq_length, :]
            
            # calculate the loss for the sequence 
            # adds a batch dimensioon at pos 0 => [1, vocab_size]
            # adds a batch dimension at pos 0 for target => [1]
            loss += criterion(prediction.unsqueeze(0), targets[j].unsqueeze(0))
        
        # avg the loss of that batch 
        loss /= batch_size
        
        # calculate grad 
        loss.backward()

        # update grad 
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        # add the loss of batch to total loss 
        total_loss += loss.item()
        if i % 100 == 0:
            print(f'Epoch [{epoch+1}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    return total_loss / len(train_loader)

def validate(val_loader, encoder, decoder, criterion):
    encoder.eval()
    decoder.eval()
    
    total_loss = 0
    
    with torch.no_grad():
        for i, (images, captions, targets, lengths) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)
            targets = targets.to(device)
            
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            
            batch_size = outputs.size(0)
            loss = 0
            for j in range(batch_size):
                seq_length = lengths[j].item() - 1
                prediction = outputs[j, seq_length, :]
                loss += criterion(prediction.unsqueeze(0), targets[j].unsqueeze(0))
            
            loss /= batch_size
            total_loss += loss.item()
    
    return total_loss / len(val_loader)