import json
from pathlib import Path

from sklearn.metrics import classification_report, accuracy_score

from main import get_train_args
from src.path import get_result_path
from src.config import model_names_list, dataset_names_list, get_dataset_display_name


def main():
    table_dir = Path("tables")
    table_dir.mkdir(exist_ok=True, parents=True)
    
    performance_dict = {}
    
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
                        for cnn_kernel_size in [3, 5, 7]:
                            if model_name != "cnn" and cnn_kernel_size > 3:
                                continue
                            
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
                                args_list.extend(["--cnn_kernel_size", str(cnn_kernel_size)])
                            
                            args = get_train_args().parse_args(args_list)
                            
                            # load result
                            result_path = get_result_path(args).with_suffix(".test.json")
                            if not result_path.exists():
                                row.append("    ")
                                continue
                            
                            with open(result_path) as f:
                                result = json.load(f)
                            y_pred = result["preds"]
                            y_true = result["targets"]
                            
                            # calculate metrics
                            if metric == "f1":
                                score = classification_report(y_true, y_pred, output_dict=True)["weighted avg"]["f1-score"]
                            elif metric == "accuracy":
                                score = accuracy_score(y_true, y_pred)
                            else:
                                raise ValueError(f"Invalid metrics: {metric}")
                            
                            performance_dict.setdefault(result_path.name, {})[metric] = score
    
    # save performance_dict
    performance_dir = Path("performance")
    performance_dir.mkdir(exist_ok=True, parents=True)
    with open(performance_dir / "performance_dict.json", "w") as f:
        json.dump(performance_dict, f, indent=4)


if __name__ == '__main__':
    main()
