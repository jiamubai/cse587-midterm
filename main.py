import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_dataset, train_model, CNNModel, RNNModel
import gensim.downloader as api
import argparse
import logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rnn', help='Model to use: cnn, rnn')
    parser.add_argument('--dataset', type=str, default='yelp', help='Dataset to use: imdb, yelp, twitter')
    parser.add_argument('--batch_size', type=int, default=1024, help='Training batch size')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--epoch', type=int, default=10, help='Training epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)

    logging.basicConfig(filename=f'result_{args.model}_{args.dataset}_epoch_{args.epoch}_lr_{args.lr}.log',
                        # format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        filemode='w')


    # Load GloVe embeddings
    embedding_model = api.load("glove-wiki-gigaword-50")
    device = 'cuda:0'
    if args.dataset == 'imdb':
        data_path = 'dataset/processed_imdb_data.csv'
        num_class = 2
    elif args.dataset == 'yelp':
        data_path = 'dataset/processed_yelp_data.csv'
        num_class = 5
    elif args.dataset == 'twitter':
        data_path = 'dataset/twitter_raw.csv'
        num_class = 3
    else:
        raise ValueError('Invalid dataset')
    train_dataset, test_dataset = load_dataset(data_path, embedding_model)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model == 'rnn':
        model = RNNModel(embedding_dim=50, hidden_dim=128, num_classes=num_class, n_layers=1).to(device)
    else:
        model = CNNModel(embedding_dim=50, num_classes=num_class, max_len=args.max_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=args.epoch)

if __name__ == '__main__':
    main()

