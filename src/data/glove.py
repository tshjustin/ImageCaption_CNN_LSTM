import os
import numpy as np
import torch
import urllib.request
import zipfile
from tqdm import tqdm

def load_glove_vectors(glove_path="./glove", embed_size=300):
    if embed_size not in [50, 100, 200, 300]:
        raise ValueError("embed_size must be one of: 50, 100, 200, 300")
        
    glove_file = os.path.join(glove_path, f'glove.6B.{embed_size}d.txt') # in .txt form, the representation is = word v1 v2 v3 v4 ... v300 => {word: [v1, ..., v300]}
    
    word2vec = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            values = line.strip().split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vectors
            
    return word2vec

def create_embedding_matrix(vocab, word2vec, embed_size=300):
    """
    The vocab that we created previously serves as our look up table. 

    Essentially, we want a mapping of word -> its embedding matrix. 

    To do so, we use a pretrained word embedding table and simply look up on the word that appears in our vocab to this matrix.
    """
    vocab_size = len(vocab)
    
    embedding_matrix = np.zeros((vocab_size, embed_size)) # (number row = vocab size, each word is represented as a embed_size vector)
    
    special_tokens = ["<pad>", "<start>", "<end>", "<unk>"]
    
    words_found = 0
    
    for word, idx in vocab.word2idx.items():
        
        if word in special_tokens:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embed_size,))
        elif word in word2vec:
            embedding_matrix[idx] = word2vec[word]
            words_found += 1
        
        # an out of domain word - init to random 
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embed_size,))
            
    return torch.FloatTensor(embedding_matrix)

def integrate_glove_embeddings(vocab, embed_size=300):
    word2vec = load_glove_vectors()
    
    embedding_weights = create_embedding_matrix(vocab, word2vec, embed_size)
    
    return embedding_weights