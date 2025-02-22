from torch.utils.data import DataLoader, Dataset
import torch
# from torchtext.data.utils import get_tokenizer
from nltk.tokenize import word_tokenize

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, embedding_model, max_len=128):
        self.texts = texts
        self.labels = labels
        self.embedding_model = embedding_model
        self.max_len = max_len
        # self.tokenizer = get_tokenizer('basic_english')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        tokens = word_tokenize(text)
        embeddings = [self.embedding_model[word] if word in self.embedding_model else [0]*50 for word in tokens]
        if len(embeddings) < self.max_len:
            embeddings.extend([[0]*50] * (self.max_len - len(embeddings)))
        else:
            embeddings = embeddings[:self.max_len]
        return {
            'embeddings': torch.tensor(embeddings, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }
