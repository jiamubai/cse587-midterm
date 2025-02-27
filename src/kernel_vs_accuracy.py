import json
from pathlib import Path

import matplotlib.pyplot as plt

from main import get_train_args
from src.path import get_result_path
from src.config import dataset_names_list, get_dataset_display_name


def main():
    figure_dir = Path("figures") / "kernel_size_vs_accuracy"
    figure_dir.mkdir(exist_ok=True, parents=True)
    
    model_name = "cnn"
    
    max_len = 512
    hidden_dim = 512
    padding = "left"
    
    # load performance dict
    with open("performance/performance_dict.json", "r") as f:
        performance_dict = json.load(f)
    
    # hidden size vs accuracy
    kernel_size_list = [3, 5, 7]
    for embed_dim in [50, 300]:
        fig = plt.figure(figsize=(5, 3))
        for dataset_name in dataset_names_list:
            accuracy_list = []
            for kernel_size in kernel_size_list:
                args = get_train_args().parse_args(
                    [
                        "--model", model_name,
                        "--dataset", dataset_name,
                        "--epoch", "20",
                        "--lr", "0.001",
                        "--embed_dim", str(embed_dim),
                        "--hidden_dim", str(hidden_dim),
                        "--max_len", str(max_len),
                        "--padding", padding,
                        "--cnn_kernel_size", str(kernel_size),
                    ]
                )
                
                # load result
                result_path = get_result_path(args).with_suffix(".test.json")
                if not result_path.exists():
                    continue
                
                accuracy = performance_dict[result_path.name]["accuracy"]
                accuracy_list.append(accuracy * 100)

            if len(kernel_size_list) != len(accuracy_list):
                continue
            plt.plot(kernel_size_list, accuracy_list,
                     marker="o", label=get_dataset_display_name[dataset_name])
        
        # save figure
        plt.xlabel("Kernel Size")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.tight_layout()
        
        figure_path = figure_dir / f"dataset={dataset_name}_embed_dim={embed_dim}.png"
        fig.savefig(figure_path)
        plt.close(fig)
            

if __name__ == '__main__':
    main()
