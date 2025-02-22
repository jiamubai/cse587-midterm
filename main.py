import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# from torchtext.datasets import IMDB, YelpReviewPolarity
# from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from utils import load_dataset, train_model, evaluate_model, CNNModel, RNNModel

# For Twitter Sentiment140, you might need to download and preprocess separately
# Assuming we have preprocessed all datasets into a standard CSV format with 'text' and 'label' columns
import pandas as pd
# from torchtext.data.utils import get_tokenizer
import gensim.downloader as api
import argparse

# Main execution (example usage)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rnn', help='Model to use: cnn, rnn')
    parser.add_argument('--dataset', type=str, default='yelp', help='Dataset to use: imdb, yelp, twitter')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--epoch', type=int, default=10, help='Training epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    print(args)


    # Load GloVe embeddings
    embedding_model = api.load("glove-wiki-gigaword-50")
    device = 'cuda:0'
    if args.dataset == 'imdb':
        data_path = 'dataset/processed_imdb_data.csv'
    elif args.dataset == 'yelp':
        data_path = 'dataset/processed_yelp_data.csv'
    elif args.dataset == 'twitter':
        data_path = 'dataset/twitter_raw.csv'
    else:
        raise ValueError('Invalid dataset')
    train_dataset, test_dataset = load_dataset(data_path, embedding_model)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model == 'rnn':
        model = RNNModel(embedding_dim=50, hidden_dim=128, num_classes=2, n_layers=1).to(device)
    else:
        model = CNNModel(embedding_dim=50, num_classes=2, max_len=args.max_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=args.epoch)

if __name__ == '__main__':
    main()

