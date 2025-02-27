from pathlib import Path


def get_result_path(args) -> Path:
    params = f'{args.model}_{args.dataset}_epoch={args.epoch}_lr={args.lr}_embed={args.embed_dim}_hidden_dim={args.hidden_dim}'
    
    if args.model == "cnn":
        params += f'_cnn_kernel_size={args.cnn_kernel_size}'
    
    return Path(f'results/result_{params}.log')
