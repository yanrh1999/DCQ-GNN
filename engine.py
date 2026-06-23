import torch
from torch.nn import functional as F
import optuna
from model import DCQ_GNN
from utils import sparse_eye
from dataset import Dataset
import copy


@torch.no_grad()
def validate(model, dataset, I):
    model.eval()
    with torch.no_grad():
        val_out = model(dataset.X, dataset.L, I)
        val_loss = F.cross_entropy(val_out[dataset.val_mask], dataset.y[dataset.val_mask]).item()
        pred = val_out[dataset.val_mask].max(dim=1)[1]
        val_acc = pred.eq(dataset.y[dataset.val_mask]).sum().item() / dataset.val_mask.sum().item()
    return val_acc, val_loss


@torch.no_grad()
def test(model, dataset, gate_save=False):
    model.eval()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    I = sparse_eye(dataset.num_nodes, device=dataset.X.device)
    L = dataset.atk_L if hasattr(dataset, 'atk_L') else dataset.L
    X = dataset.atk_X if hasattr(dataset, 'atk_X') else dataset.X
    start_event.record()
    out = model(X, L, I, gate_save=gate_save)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)

    test_loss = F.cross_entropy(out[dataset.test_mask], dataset.y[dataset.test_mask]).item()
    pred = out[dataset.test_mask].max(dim=1)[1]
    acc = pred.eq(dataset.y[dataset.test_mask]).sum().item() / dataset.test_mask.sum().item()
    return acc, elapsed_time_ms, test_loss


def train(model, optimizer, dataset, epochs, patience=100):
    best_val_acc = 0.0
    best_val_loss = float('inf')
    I = sparse_eye(dataset.num_nodes, device=dataset.X.device)
    count = 0
    best_model = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(dataset.X, dataset.L, I)
        loss = F.cross_entropy(out[dataset.train_mask], dataset.y[dataset.train_mask])
        loss.backward()
        optimizer.step()

        val_acc, val_loss = validate(model, dataset, I)

        if (epoch + 1) % 50 == 0:
            print("Epoch: {}, Training Loss: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.2f}%".format(
                epoch + 1, loss.item(), val_loss, val_acc * 100))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            count = 0
        else:
            count += 1

        if count == patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
        
    model.load_state_dict(best_model)
    return model


def objective(trial, args, device):
    """Optuna objective function for hyperparameter search."""
    feature_transform = trial.suggest_categorical('feature transform', [True, False])
    dataset = Dataset(name=args.dataset, device=device, train_ratio=args.train_ratio,
                      normaliation=feature_transform)
    I = sparse_eye(dataset.num_nodes, device=dataset.X.device)

    num_layer = trial.suggest_int('num layer', 1, 3)
    dr = trial.suggest_float('dr', 0.0, 0.9)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd = trial.suggest_float('wd', 1e-6, 1e-2, log=True)

    low_pass_opt = []
    high_pass_opt = []
    for layer in range(num_layer):
        l_opt = trial.suggest_categorical(f'low-pass opt {layer + 1}', ['cvx', 'ccv'])
        h_opt = trial.suggest_categorical(f'high-pass opt {layer + 1}', ['cvx', 'ccv'])
        low_pass_opt.append(l_opt)
        high_pass_opt.append(h_opt)

    model = DCQ_GNN(num_layer=num_layer, in_features=dataset.X.shape[1],
                    hidden_features=args.hidden_features,
                    out_features=dataset.num_classes, dr=dr,
                    opt=[low_pass_opt, high_pass_opt], multi_scale=False).to(dataset.X.device)

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
        {'params': decay_params, 'weight_decay': wd}
    ], lr=lr)

    best_trial_val_acc = 0.0

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        model.train()
        out = model(dataset.X, dataset.L, I)
        loss = F.cross_entropy(out[dataset.train_mask], dataset.y[dataset.train_mask])
        loss.backward()
        optimizer.step()

        val_acc, _ = validate(model, dataset, I)
        best_trial_val_acc = max(best_trial_val_acc, val_acc)

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_trial_val_acc
