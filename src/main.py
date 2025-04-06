import os
import torch
import wandb
from PIL import Image
from dotenv import load_dotenv
from train import train, validate
from model.lstm import LSTMDecoder
from torch.utils.data import DataLoader
from utils import generate_caption, device
import torchvision.transforms as transforms
from model.encode_visuals import EncoderCNN
from data.data_process import clean_captions, split_dataset
from data.data_loader import create_vocabulary, create_data_loaders, FlickrPartialSequenceDataset, collate_fn

load_dotenv()

def setup_wandb():
    wandb_api_key = os.environ.get("WANDB_API_KEY")

    wandb.login(key=wandb_api_key)
    wandb.init(project="image-captioning-cnn-lstm", name="training-run")

    config = wandb.config
    config.learning_rate = 3e-4
    config.batch_size = 64
    config.epochs = 10
    config.embed_size = 256
    config.hidden_size = 512
    config.num_layers = 1


def main():

    setup_wandb()

    embed_size = 300
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
        from data.glove import integrate_glove_embeddings
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
    
    criterion = torch.nn.CrossEntropyLoss()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    
    print(f"Starting training for {num_epochs} epochs")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss = train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch)
        
        val_loss = validate(val_loader, encoder, decoder, criterion)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
            }, 'model_best.pth')

    final_model_path = 'model_final.pth'
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'encoder_optimizer': encoder_optimizer.state_dict(),
        'decoder_optimizer': decoder_optimizer.state_dict(),
    }, final_model_path)
    
    print(f"Final model saved to {final_model_path}")
    
    
    # eval
    test_dataset = FlickrPartialSequenceDataset(test_data, vocab, transform)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    encoder.eval()
    decoder.eval()
    
    sample_count = 0
    samples_seen = set()
    
    # check for actual generation 
    for images, captions, targets, lengths in test_dataloader:
        image_path = test_dataset.image_paths[0]  
        
        image_name = os.path.basename(image_path)
        if image_name in samples_seen:
            continue
        
        samples_seen.add(image_name)
        for item in test_data:
            if item[0] == image_name:
                actual_caption = item[1]
                break
        
        generated_caption = generate_caption(encoder, decoder, images[0].unsqueeze(0), vocab)
        
        print(f"\nSample {sample_count+1}:")
        print(f"Image: {image_name}")
        print(f"Actual caption: {actual_caption}")
        print(f"Generated caption: {generated_caption}")


        wandb.log({
            f"test_sample_{sample_count+1}": wandb.Image(
                Image.open(image_path).convert('RGB'),
                caption=f"Actual: {actual_caption}\nGenerated: {generated_caption}"
            )
        })

        sample_count += 1
        if sample_count >= 5:
            break
    
    wandb.finish()

if __name__ == "__main__":
    main()