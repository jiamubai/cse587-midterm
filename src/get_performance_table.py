import json
from pathlib import Path

from sklearn.metrics import classification_report, accuracy_score

from main import get_train_args
from src.path import get_result_path
from src.config import model_names_list, dataset_names_list, get_dataset_display_name


def main():
    table_dir = Path("tables")
    table_dir.mkdir(exist_ok=True, parents=True)
    
    # get performance_dict
    with open("performance/performance_dict.json", "r") as f:
        performance_dict = json.load(f)

    max_len = 512
    padding = "left"
    
    
    for metric in ["f1", "accuracy"]:
        for hidden_dim in [128, 256, 512]:
            for embed_dim in [50, 300]:
                table = []
                first_row = ["Dataset"] + [model_name.upper() for model_name in model_names_list]
                table.append(first_row)
                
                for dataset_name in dataset_names_list:
                    row = [f"{get_dataset_display_name[dataset_name]:7s}"]
                    for model_name in model_names_list:
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
                            row.append("    ")
                            continue
                        
                        score = performance_dict[result_path.name][metric]
                        row.append(f"{score*100:.1f}")
                    table.append(row)
                
                table_path = table_dir / f"metric={metric}_embed_dim={embed_dim}_hidden_dim={hidden_dim}.txt"
                with open(table_path, "w") as f:
                    for row in table:
                        f.write(" & ".join(row) + r" \\")
                        f.write("\n")
                    f.write("\n")


if __name__ == '__main__':
    main()
