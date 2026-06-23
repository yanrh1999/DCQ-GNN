import yaml
import os
import warnings
import torch
import random
from deeprobust.graph.global_attack import Metattack
from torch_geometric.utils import to_scipy_sparse_matrix, dense_to_sparse


def load_config(config_file=None, dataset=None):
    """Load YAML config for a given dataset, falling back to defaults."""
    base_path = './config/'
    dataset_config = os.path.join(base_path, f'{dataset}.yaml')
    default_config = os.path.join(base_path, 'default config.yaml')
    if config_file:
        target_path = os.path.join(base_path, config_file)
    elif os.path.isfile(dataset_config):
        target_path = dataset_config
    elif os.path.isfile(default_config):
        warnings.warn("No config file provided. Using 'default config.yaml'")
        target_path = default_config
    else:
        raise FileNotFoundError("'default config.yaml' missed")

    if os.path.isfile(target_path):
        with open(target_path, 'r') as f:
            config = yaml.safe_load(f)
            print(f"Using config: {os.path.split(target_path)[-1]}")
        return config
    else:
        raise FileNotFoundError(f"No config file found at {target_path}. Please provide a valid config file.")


def sparse_eye(num_nodes, device=None, dtype=torch.float):
    """Returns a sparse identity matrix of shape (num_nodes, num_nodes)."""
    idx = torch.arange(num_nodes, device=device)
    indices = torch.stack([idx, idx], dim=0)
    values = torch.ones(num_nodes, device=device, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes)).coalesce()


def dice_attack(edge_index, num_nodes, y, ratio):
    """DICE attack: deletes intra-class edges and adds inter-class edges."""
    num_edges = edge_index.shape[1]
    n_perturb = int(ratio * num_edges)

    row, col = edge_index
    mask_same_class = (y[row] == y[col])
    candidates_for_deletion = mask_same_class.nonzero(as_tuple=True)[0]

    if len(candidates_for_deletion) < n_perturb:
        indices_to_remove = candidates_for_deletion
    else:
        perm = torch.randperm(len(candidates_for_deletion))
        indices_to_remove = candidates_for_deletion[perm[:n_perturb]]

    mask_keep = torch.ones(num_edges, dtype=torch.bool)
    mask_keep[indices_to_remove] = False
    current_edge_index = edge_index[:, mask_keep]

    existing_edges = set(zip(current_edge_index[0].tolist(), current_edge_index[1].tolist()))
    new_edges_list = []
    added_count = 0

    while added_count < n_perturb:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v and (u, v) not in existing_edges and (v, u) not in existing_edges:
            if y[u] != y[v]:
                new_edges_list.append([u, v])
                existing_edges.add((u, v))
                added_count += 1

    if new_edges_list:
        new_edges = torch.tensor(new_edges_list).t()
        final_edge_index = torch.cat([current_edge_index, new_edges], dim=1)
    else:
        final_edge_index = current_edge_index

    return final_edge_index


def metattack(data, ptb_rate=0.1):
    """Metaattack via DeepRobust: poisons the graph adjacency to degrade GNN performance."""
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
    features = data.x.cpu().numpy()
    labels = data.y.cpu().numpy()

    from deeprobust.graph.defense import GCN
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                    nhid=16, dropout=0.5, with_relu=False, with_bias=False, device='cuda')
    surrogate = surrogate.to('cuda')
    surrogate.fit(features, adj, labels, data.train_mask.cpu().numpy())

    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                      attack_structure=True, attack_features=False, device='cuda', lambda_=0)
    model = model.to('cuda')

    n_perturbations = int(ptb_rate * (adj.sum() // 2))
    model.attack(features, adj, labels, data.train_mask.cpu().numpy(),
                 data.val_mask.cpu().numpy(), n_perturbations, ll_constraint=False)

    poisoned_adj = model.modified_adj
    poisoned_edge_index, _ = dense_to_sparse(poisoned_adj)
    torch.save(poisoned_edge_index, "metaattack.pt")
    return poisoned_edge_index.to(data.edge_index.device)


def node_perturbation(x, ratio, noise_scale=1.0):
    """Injects Gaussian noise into a fraction of node features, scaled by global std."""
    device = x.device
    num_nodes, num_features = x.shape
    n_perturb = int(ratio * num_nodes)

    if n_perturb == 0:
        return x.clone()

    perm = torch.randperm(num_nodes, device=device)
    target_nodes = perm[:n_perturb]

    global_std = x.std(dim=0, keepdim=True) + 1e-8

    noisy_x = x.clone()
    raw_noise = torch.randn(n_perturb, num_features, device=device)
    scaled_noise = raw_noise * global_std * noise_scale
    noisy_x[target_nodes] += scaled_noise

    return noisy_x
