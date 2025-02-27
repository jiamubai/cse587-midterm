import json
from pathlib import Path

import matplotlib.pyplot as plt

from main import get_train_args
from src.path import get_result_path
from src.config import model_names_list, dataset_names_list


def main():
    figure_dir = Path("figures") / "hidden_size_vs_accuracy"
    figure_dir.mkdir(exist_ok=True, parents=True)
    
    max_len = 512
    padding = "left"
    
    # load performance dict
    with open("performance/performance_dict.json", "r") as f:
        performance_dict = json.load(f)
    
    # hidden size vs accuracy
    hidden_dim_list = [128, 256, 512]
    for embed_dim in [50, 300]:
        for dataset_name in dataset_names_list:
            fig = plt.figure(figsize=(5, 3))
            for model_name in model_names_list:
                accuracy_list = []
                for hidden_dim in hidden_dim_list:
                    args_list = [
                        "--model", model_name,
                        "--dataset", dataset_name,
                        "--epoch", "20",
                        "--lr", "0.001",
                        "--embed_dim", str(embed_dim),
                        "--hidden_dim", str(hidden_dim),
                        "--max_len", str(max_len),
                        "--padding", padding,
                    ]
                    if model_name == "cnn":
                        args_list.extend(["--cnn_kernel_size", "7"])
                    
                    args = get_train_args().parse_args(args_list)
                    
                    # load result
                    result_path = get_result_path(args).with_suffix(".test.json")
                    if not result_path.exists():
                        continue
                    
                    accuracy = performance_dict[result_path.name]["accuracy"]
                    accuracy_list.append(accuracy * 100)

                if len(hidden_dim_list) != len(accuracy_list):
                    continue
                plt.plot(hidden_dim_list, accuracy_list,
                         marker="o", label=model_name.upper())
            
            # save figure
            plt.xlabel("Hidden Size")
            plt.ylabel("Accuracy (%)")
            plt.legend()
            plt.tight_layout()
            
            figure_path = figure_dir / f"dataset={dataset_name}_embed_dim={embed_dim}.png"
            fig.savefig(figure_path)
            plt.close(fig)
            

if __name__ == '__main__':
    main()
