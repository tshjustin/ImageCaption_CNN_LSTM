import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_caption(encoder, decoder, image_tensor, vocab, max_length=20):
    with torch.no_grad():
        features = encoder(image_tensor.to(device))

        states = None

        start_token = torch.tensor([1]).to(device)  
        inputs = encoder.embed(features).unsqueeze(1)
        
        sampled_ids = []
        for i in range(max_length):
            hiddens, states = decoder.lstm(inputs, states)
            outputs = decoder.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            
            sampled_ids.append(predicted.item())
            
            # Break if <end> token is predicted
            if predicted.item() == 2:  # <end> token index is 2
                break
                
            inputs = decoder.embed(predicted).unsqueeze(1)
    
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        
        if word == "<end>":
            break
        if word not in ["<start>", "<pad>"]:
            sampled_caption.append(word)
    
    return ' '.join(sampled_caption)