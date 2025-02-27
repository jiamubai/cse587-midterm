from pathlib import Path

import matplotlib.pyplot as plt
import torch

from utils import load_dataset, get_dataset_path, get_embedding_model


def get_text_length(dataset, padding: str = "left") -> list[int]:
    if padding != "left":
        raise NotImplementedError(f"Padding '{padding}' is not implemented.")
    
    text_len = []
    for data in dataset:
        padding_num = 0
        for embed in data["embeddings"]:
            embed: torch.Tensor = embed
            if embed.sum().item() == 0:
                padding_num += 1
            else:
                break
        text_len.append(len(data["embeddings"]) - padding_num)
    
    return text_len


def main():
    stat_figure_dir = Path("figures/dataset_stat")
    stat_figure_dir.mkdir(exist_ok=True, parents=True)
    
    embed_dim = 50
    max_len = 512
    padding = "left"
    
    for dataset_name in ["imdb", "yelp", "twitter"]:
        print(f"Dataset: {dataset_name}")
        
        data_path, _ = get_dataset_path(dataset_name)
        embedding_model = get_embedding_model(embed_dim=embed_dim)
        _, test_dataset = load_dataset(data_path, embedding_model, max_len=max_len, padding=padding, embed_dim=embed_dim)
        
        text_len = get_text_length(test_dataset, padding=padding)
        
        # histogram
        fig = plt.figure(figsize=(5, 3))
        plt.hist(text_len, bins=50)
        
        if dataset_name == "twitter":
            plt.xlim(0, 64)
        else:
            plt.xlim(0, 512)
        
        plt.xlabel("Text Length")
        plt.ylabel("Frequency")
        plt.tight_layout()
        
        plt.savefig(stat_figure_dir / f"{dataset_name}_text_length_hist.png")
        plt.close(fig)
        

if __name__ == '__main__':
    main()
