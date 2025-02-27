from pathlib import Path
import json

import pandas as pd
from .dataset import SentimentDataset
from tqdm import tqdm
import torch
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import logging
from nltk.tokenize import word_tokenize
import gensim.downloader as api


def get_embedding_model(embed_dim):
    # Load GloVe embeddings
    if embed_dim == 50:
        embedding_model = api.load("glove-wiki-gigaword-50")
    elif embed_dim == 300:
        embedding_model = api.load("glove-wiki-gigaword-300")
    else:
        raise ValueError('Invalid embedding dimension')
    return embedding_model


def get_dataset_path(dataset_name) -> tuple[str, int]:
    if dataset_name == 'imdb':
        data_path = 'dataset/processed_imdb_data.csv'
        num_class = 2
    elif dataset_name == 'yelp':
        data_path = 'dataset/processed_yelp_data.csv'
        num_class = 5
    elif dataset_name == 'twitter':
        data_path = 'dataset/twitter_raw.csv'
        num_class = 2
    else:
        raise ValueError('Invalid dataset')
    return data_path, num_class


# Load and preprocess datasets
def load_dataset(file_path, embedding_model, max_len=128, test_size=0.2, padding='right', embed_dim=50):
    if 'imdb' in file_path:
        df = pd.read_csv(file_path)
        df = df[(df['sentiment_id'] <= 3)|(df['sentiment_id'] >=8)]
        texts = df['review'].tolist()
        labels = df['sentiment'].tolist()
        labels = [0 if label == 'negative' else 1 for label in labels]
    elif 'yelp' in file_path:
        df = pd.read_csv(file_path)
        texts = df['review'].tolist()
        labels = df['stars'].tolist()
        labels = [label - 1 for label in labels]
        # df = df[df['stars'] != 3]
    else:
        df = pd.read_csv(file_path, encoding="ISO-8859-1", engine="python", on_bad_lines='skip')
        df.columns = ["sentiment", "time", "date", "query", "username", "review"]
        df = df.sample(n=10000)
        texts = df['review'].tolist()
        labels = df['sentiment'].tolist()
        labels = [label // 4 for label in labels]

    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=test_size, random_state=42)
    train_dataset = SentimentDataset(train_texts, train_labels, embedding_model, max_len, padding, embed_dim)
    test_dataset = SentimentDataset(test_texts, test_labels, embedding_model, max_len, padding, embed_dim)
    return train_dataset, test_dataset


def train_model(model, train_loader, test_loader, criterion, optimizer, device, epochs=10, result_path: Path = None):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc='Training'):
            embeddings = batch['embeddings'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Training Loss: {total_loss / len(train_loader)}')
        logging.info(f'Training Loss: {total_loss / len(train_loader)}')

        print('Evaluating on Training Set...')
        logging.info('Evaluating on Training Set...')

        train_report, train_accuracy = evaluate_model(model, train_loader, device)
        print(train_report)
        logging.info(train_report)
        print(f'Training Accuracy: {train_accuracy}')
        logging.info(f'Training Accuracy: {train_accuracy}')

        print('Evaluating on Test Set...')
        logging.info('Evaluating on Test Set...')

        test_output_path = result_path.with_suffix(".test.json")
        test_report, test_accuracy = evaluate_model(model, test_loader, device, result_path=test_output_path)
        print(test_report)
        logging.info(test_report)
        print(f'Test Accuracy: {test_accuracy}')
        logging.info(f'Test Accuracy: {test_accuracy}')

@torch.no_grad()
def evaluate_model(model, dataloader, device, result_path: Path = None):
    model.eval()
    preds, targets = [], []
    for batch in tqdm(dataloader, desc='Evaluating'):
        embeddings = batch['embeddings'].to(device)
        labels = batch['label'].to(device)
        outputs = model(embeddings)
        preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
        targets.extend(labels.cpu().numpy().tolist())
    
    if result_path is not None:
        with open(result_path, 'w') as f:
            json.dump({'preds': preds, 'targets': targets}, f)
    
    return classification_report(targets, preds), accuracy_score(targets, preds)
