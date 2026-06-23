import torch
from model import DCQ_GNN
from engine import train, test, objective
from dataset import Dataset
from argparse import ArgumentParser
import yaml
import numpy as np
import optuna
from optuna.trial import TrialState
import torch_geometric.transforms as T
from utils import load_config
import random


def set_seed(seed):
    """Locks all sources of randomness for reproducible GNN evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = ArgumentParser(description="DCQ-GNN: Dual-Channel Quadratic Graph Neural Network")
    parser.add_argument('dataset', type=str, help='Dataset name')
    parser.add_argument('-m', '--mode', choices=['train', 'search'], type=str,
                        help="train: train a new model by config file\n"
                             "search: search best hyperparameters",
                        default='train')
    parser.add_argument('-n', '--num_run', type=int, default=10,
                        help='Number of runs for averaging results under "train" mode')
    parser.add_argument('--config_file', type=str, default=None, help='Path of the config file')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Train ratio for dataset split')
    parser.add_argument('--hidden_features', type=int, default=64, help='Number of hidden features')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--atk', choices=['DICE', 'Metaattack', 'Noise'], type=str,
                        default='None', help='Attack selection')
    parser.add_argument('--ptb_rate', type=float, default=0.1, help="Perturb rate of attack")
    parser.add_argument('--device', type=int, default=0, help='GPU device ID, default is 0')

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'train':
        config = load_config(config_file=args.config_file, dataset=args.dataset)
        acc_list = []
        time_list = []

        dataset = Dataset(name=args.dataset, device=device, train_ratio=args.train_ratio,
                          normaliation=config['feature transform'],
                          atk=args.atk, ptb_rate=args.ptb_rate)
        for i in range(args.num_run):
            print(f'\n--- Run {i + 1}/{args.num_run} ---')
            set_seed(i)

            dataset.resplit()

            low_pass_opt = []
            high_pass_opt = []
            for layer in range(config['num layer']):
                low_pass_opt.append(config[f'low-pass opt {layer + 1}'])
                high_pass_opt.append(config[f'high-pass opt {layer + 1}'])

            model = DCQ_GNN(num_layer=config['num layer'], in_features=dataset.X.shape[1],
                            hidden_features=args.hidden_features,
                            out_features=dataset.num_classes, dr=config['dr'],
                            opt=[low_pass_opt, high_pass_opt], multi_scale=False).to(device)

            if i == 0:
                total_params = sum(p.numel() for p in model.parameters())
                print(f"Total Parameters: {total_params}")

            filter_params, decay_params, no_decay_params = [], [], []
            for name, param in model.named_parameters():
                if 'lambda_t' in name or 'beta' in name:
                    filter_params.append(param)
                elif 'bias' in name or 'norm' in name.lower():
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            optimizer = torch.optim.Adam([
                {'params': filter_params, 'weight_decay': 0.0},
                {'params': no_decay_params, 'weight_decay': 0.0},
                {'params': decay_params, 'weight_decay': config['wd']}
            ], lr=config['lr'])

            train(model=model, optimizer=optimizer, dataset=dataset, epochs=args.epochs)
            acc, time_cost, _ = test(model, dataset, gate_save=False)
            acc_list.append(acc)
            time_list.append(time_cost)

        avg_acc = np.mean(acc_list)
        std = np.std(acc_list)
        avg_time_cost = np.mean(time_list)
        print(f'\n====================================')
        print(f'Average Test Accuracy ({args.num_run} runs): {avg_acc * 100:.2f}±{std * 100:.2f}%')
        print(f'Average Inference Time: {avg_time_cost:.3f} ms')
        print(f'====================================\n')
        with open("results.txt", "a") as f:
            f.write(f"{args.dataset} \t {avg_acc * 100:.2f} \\pm {std * 100:.2f}\n")

    elif args.mode == 'search':
        study = optuna.create_study(direction="maximize",
                                    pruner=optuna.pruners.MedianPruner(
                                        n_startup_trials=10, n_warmup_steps=50, interval_steps=10))
        study.optimize(lambda trial: objective(trial, args, device), n_trials=100, timeout=3600)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)

        with open(f'./config/{args.dataset}.yaml', 'w') as f:
            print("  Params: ")
            yaml.dump(trial.params, f)
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
