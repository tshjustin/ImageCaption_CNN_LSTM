import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5, pretrained_embeddings=None):
        super(LSTMDecoder, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        if pretrained_embeddings is not None:
            self.embed.weight = nn.Parameter(pretrained_embeddings)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, captions, lengths=None, max_length=20) :
        embeddings = self.embed(captions)
        features = features.unsqueeze(1)
        
        if lengths is not None:
            embeddings = torch.cat((features, embeddings), 1)
            packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
            packed_outputs, _ = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
            outputs = self.dropout(outputs)
            outputs = self.linear(outputs)
            
            return outputs
        else:
            states = None
            outputs = []
            
            inputs = features
            
            for i in range(max_length):
                hiddens, states = self.lstm(inputs, states)
                outputs.append(self.linear(hiddens.squeeze(1)))
                inputs = self.embed(outputs[-1].argmax(1)).unsqueeze(1)
                
            return torch.stack(outputs, 1)
        
    def sample(self, features, states=None, max_len=20, start_token=0, end_token=1):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())
            
            if predicted.item() == end_token:
                break
                
            inputs = self.embed(predicted).unsqueeze(1)
            
        return sampled_ids