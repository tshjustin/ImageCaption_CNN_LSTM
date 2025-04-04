import os
import numpy as np
import torch
import urllib.request
import zipfile
from tqdm import tqdm

def download_glove(glove_dir='./glove', glove_name='glove.6B.zip'):
    os.makedirs(glove_dir, exist_ok=True)
    
    glove_url = 'https://nlp.stanford.edu/data/glove.6B.zip'
    
    zip_path = os.path.join(glove_dir, glove_name)
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(glove_url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(glove_dir)
            
    return glove_dir

def load_glove_vectors(glove_path, embed_size=300):
    if embed_size not in [50, 100, 200, 300]:
        raise ValueError("embed_size must be one of: 50, 100, 200, 300")
        
    glove_file = os.path.join(glove_path, f'glove.6B.{embed_size}d.txt')
    
    word2vec = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            values = line.strip().split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vectors
            
    return word2vec

def create_embedding_matrix(vocab, word2vec, embed_size=300):
    vocab_size = len(vocab)
    
    embedding_matrix = np.zeros((vocab_size, embed_size))
    
    special_tokens = ["<pad>", "<start>", "<end>", "<unk>"]
    
    words_found = 0
    
    for word, idx in vocab.word2idx.items():
        if word in special_tokens:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embed_size,))
        elif word in word2vec:
            embedding_matrix[idx] = word2vec[word]
            words_found += 1
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embed_size,))
            
    return torch.FloatTensor(embedding_matrix)

def integrate_glove_embeddings(vocab, embed_size=300, trainable=False):
    glove_dir = download_glove()
    word2vec = load_glove_vectors(glove_dir, embed_size)
    
    embedding_weights = create_embedding_matrix(vocab, word2vec, embed_size)
    
    return embedding_weights