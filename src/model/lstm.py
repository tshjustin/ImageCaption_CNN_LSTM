import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5, pretrained_embeddings=None):
        """
        init just sets up the architecture, the acutal using part goes in the forward
        """
        super(LSTMDecoder, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embed_size) # creates embedding layer that converts token indices to dense vector -> Shapee of (vocab_size, embed_size (how each word is represented))
        
        if pretrained_embeddings is not None:
            self.embed.weight = nn.Parameter(pretrained_embeddings) # wraps in parameter object such that its accounted inside the whole networks params 
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0) # creates the archtitecture 
        self.linear = nn.Linear(hidden_size, vocab_size) # project the hidden state to the vocab probabilities
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, captions, lengths=None, max_length=20):
        """
        From the dataloader, from n samples, we create n-1 sequence-target pairs => n * (n-1) total samples. 

        For each of these samples, they have a target which is the next word of the sequence 

        To train the LSTM, we perform the following: 

        1. Embed the caption (that is of different length each time)

        2. Encode the iamge and treat it as our first "word"

        3. concatenate the image and caption such that we get [caption, image] (along axis = 1)

        4. pack_padded 

        5. pad_packed (these 2 operations are much harder to understand)
        """
        # convert indices to embeddings 
        embeddings = self.embed(captions)
        
        # add sequence dimension => [batch_size, feature_dim] -> [batch_size, 1, feature_dim]
        # this is needed since LSTM requires [batch_size, sequence_length, feature_dim]
        features = features.unsqueeze(1)

        
        if lengths is not None:

            # concatenate image features with word embeddings, column-wise (row-operation) - [batch_size, max_seq_length+1, embed_dim]
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
"""
data = [
    Image(1), Image(2),           # t=0: all sequences have tokens
    "<start>"(1), "<start>"(2),   # t=1: all sequences have tokens
    "dog"(1), "cat"(2),           # t=2: all sequences have tokens
    "sleeping"(2)                 # t=3: only sequence 2 has a token
]

outputs = [
    [lstm_out(Image1), lstm_out("<start>"1), lstm_out("dog"1), 0],
    [lstm_out(Image2), lstm_out("<start>"2), lstm_out("cat"2), lstm_out("sleeping"2)]
]

outputs = [
    [hidden_state(t=0,seq=1), hidden_state(t=1,seq=1), hidden_state(t=2,seq=1), 0],
    [hidden_state(t=0,seq=2), hidden_state(t=1,seq=2), hidden_state(t=2,seq=2), hidden_state(t=3,seq=2)]
]
"""

