import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import load_dataset, get_dataset_path, get_embedding_model, train_model, CNNModel, RNNModel, LSTMModel
from src.path import get_result_path


def get_train_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', help='Model to use: cnn, rnn, lstm')
    parser.add_argument('--dataset', type=str, default='twitter', help='Dataset to use: imdb, yelp, twitter')
    parser.add_argument('--batch_size', type=int, default=1024, help='Training batch size')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Size of hidden states')
    parser.add_argument('--epoch', type=int, default=20, help='Training epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--rnn_layer', type=int, default=1, help='')
    parser.add_argument('--cnn_kernel_size', type=int, default=5, help='')
    parser.add_argument('--embed_dim', type=int, default=50, help='')
    parser.add_argument('--padding', type=str, default='left', help='')
    
    return parser


def main():
    parser = get_train_args()
    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    # random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    # np.random.seed(seed)

    result_path = get_result_path(args)

    logging.basicConfig(
        filename=result_path,
        # format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        filemode='w'
    )

    embedding_model = get_embedding_model(embed_dim=args.embed_dim)
    
    device = 'cuda:0'
    
    data_path, num_class = get_dataset_path(args.dataset)
    
    train_dataset, test_dataset = load_dataset(data_path, embedding_model, max_len=args.max_len, padding=args.padding, embed_dim=args.embed_dim)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.model == 'rnn':
        model = RNNModel(embedding_dim=args.embed_dim, hidden_dim=args.hidden_dim, num_classes=num_class, n_layers=args.rnn_layer).to(device)
    elif args.model == 'lstm':
        model = LSTMModel(embedding_dim=args.embed_dim, hidden_dim=args.hidden_dim, num_classes=num_class, n_layers=args.rnn_layer).to(device)
    elif args.model == 'cnn':
        model = CNNModel(embedding_dim=args.embed_dim, hidden_dim=args.hidden_dim, num_classes=num_class, max_len=args.max_len, kernel_size=args.cnn_kernel_size).to(device)
    else:
        raise ValueError(f'Invalid model: {args.model}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=args.epoch, result_path=result_path)

if __name__ == '__main__':
    main()

