import json
from pathlib import Path

import matplotlib.pyplot as plt

from main import get_train_args
from utils import load_dataset, get_dataset_path, get_embedding_model
from src.path import get_result_path
from src.get_dataset_stat import get_text_length


def get_accuracy_for_different_length(text_len, result, dataset_name: str) -> dict[int, float]:
    if dataset_name == "twitter":
        length_thresholds = [0, 8, 16, 24, 32, 40, 48, 56, 64]
    else:
        length_thresholds = [0, 64, 128, 192, 256, 320, 384, 448, 511]
    
    accuracy_for_different_length = {}
    for threshold_idx, lower_bound in enumerate(length_thresholds[:-1]):
        correct_count = 0
        all_count = 0
        for data_idx, length in enumerate(text_len):
            if length > lower_bound and length <= length_thresholds[threshold_idx + 1]:
                all_count += 1
                if result["preds"][data_idx] == result["targets"][data_idx]:
                    correct_count += 1
        if all_count > 10:
            accuracy_for_different_length[
                (lower_bound + length_thresholds[threshold_idx + 1]) // 2
            ] = correct_count / all_count
    
    return accuracy_for_different_length


def main():
    figures_dir = Path("figures/len_vs_accuracy")
    figures_dir.mkdir(exist_ok=True, parents=True)
    
    embed_dim = 50
    padding = "left"
    max_len = 512

    for dataset_name in ["imdb", "yelp", "twitter"]:
        # load dataset
        data_path, _ = get_dataset_path(dataset_name)
        embedding_model = get_embedding_model(embed_dim=embed_dim)
        _, test_dataset = load_dataset(
            data_path, embedding_model, max_len=max_len,
            padding=padding, embed_dim=embed_dim
        )
        
        # get text length
        text_len = get_text_length(test_dataset, padding=padding)
        
        fig = plt.figure(figsize=(10, 5))
        for model_name in ["cnn", "rnn", "lstm"]:
            print(f"Model: {model_name}, Dataset: {dataset_name}")
            # load result
            args = get_train_args().parse_args(
                [
                    "--model", model_name,
                    "--dataset", dataset_name,
                    "--epoch", "20",
                    "--lr", "0.001",
                    "--embed_dim", str(embed_dim),
                    "--hidden_dim", "512",
                    "--max_len", str(max_len),
                    "--padding", padding,
                ]
            )
            
            result_path = get_result_path(args).with_suffix(".test.json")
            if not result_path.exists():
                continue
            
            with open(result_path) as f:
                result = json.load(f)
            
            # get accuracy for different length
            accuracy_for_different_length = get_accuracy_for_different_length(
                text_len, result, dataset_name=dataset_name
            )
            
            # plot
            plt.plot(
                list(accuracy_for_different_length.keys()),
                list(accuracy_for_different_length.values()),
                label=model_name.upper(),
                marker="o",
            )
        
        if dataset_name == "twitter":
            plt.xlim(0, 64)
        else:
            plt.xlim(0, 512)
        
        plt.xlabel("Text Length")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        
        figure_path = figures_dir / f"{dataset_name}.png"
        plt.savefig(figure_path)
        plt.close(fig)


if __name__ == '__main__':
    main()
