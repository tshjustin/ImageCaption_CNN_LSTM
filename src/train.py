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
    
    total_loss = 0
    
    for i, (images, captions, targets, lengths) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)
        targets = targets.to(device)
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        
        batch_size = outputs.size(0)
        
        loss = 0
        
        # for each sample in the batch
        for j in range(batch_size):
            seq_length = lengths[j].item() - 1
            prediction = outputs[j, seq_length, :]
            
            loss += criterion(prediction.unsqueeze(0), targets[j].unsqueeze(0))
        
        loss /= batch_size
        
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        
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

def main():
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 10
    batch_size = 64
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    caption_data = clean_captions()
    train_data, test_data = split_dataset(caption_data, split_ratio=0.8)
    
    vocab = create_vocabulary(caption_data)
    vocab_size = len(vocab)
    
    use_glove = True
    pretrained_embeddings = None
    
    if use_glove:
        pretrained_embeddings = integrate_glove_embeddings(vocab, embed_size=embed_size, trainable=True)
    
    train_loader, val_loader = create_data_loaders(
        train_data, test_data, vocab, transform, batch_size
    )
    
    encoder = EncoderCNN(embed_size).to(device)
    decoder = LSTMDecoder(
        embed_size, 
        hidden_size, 
        vocab_size, 
        num_layers,
        pretrained_embeddings=pretrained_embeddings
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch)
        
        val_loss = validate(val_loader, encoder, decoder, criterion)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
            }, 'model_best.pth')
    

if __name__ == '__main__':
    main()