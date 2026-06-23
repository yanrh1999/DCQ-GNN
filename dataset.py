import torch
from torch_geometric.datasets import Amazon, Planetoid, Actor, Coauthor, HeterophilousGraphDataset, \
    WikipediaNetwork, WebKB, WikiCS
from torch_geometric.utils import to_torch_coo_tensor, get_laplacian
import random
from torch_geometric import transforms as T
from utils import dice_attack, metattack, node_perturbation


class Dataset:
    """Unified dataset loader for node classification benchmarks.

    Supports homophilic (Cora, CiteSeer, PubMed, etc.) and heterophilic
    (chameleon, Squirrel, Actor, etc.) graphs. Optionally applies
    adversarial attacks (DICE, Metaattack, Noise) to the graph.
    """
    def __init__(self, name='Computers', device=None, train_ratio=0.6,
                 normaliation=True, atk=None, ptb_rate=0.1):
        self.device = device
        root_path = '../../dataset'
        if name in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(root=root_path, name=name)
        elif name in ['CS', 'Physics']:
            dataset = Coauthor(root=root_path, name=name)
        elif name in ['Computers', 'Photo']:
            dataset = Amazon(root=root_path, name=name)
        elif name in ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]:
            dataset = HeterophilousGraphDataset(root=root_path, name=name)
        elif name in ["chameleon", "crocodile", "Squirrel"]:
            dataset = WikipediaNetwork(root=root_path, name=name)
        elif name in ["Cornell", "Texas", "Wisconsin"]:
            dataset = WebKB(root=root_path, name=name)
        elif name == 'WikiCS':
            dataset = WikiCS(root=f'{root_path}/WikiCS')
        elif name == 'Actor':
            dataset = Actor(root=f'{root_path}/Actor')
        else:
            raise ValueError(f"Dataset {name} not recognized.")

        self.data = dataset[0]
        transform = T.RandomNodeSplit(num_val=(1 - train_ratio) / 2, num_test=(1 - train_ratio) / 2)
        if normaliation:
            transform = T.Compose([transform, T.NormalizeFeatures()])

        data = transform(self.data)
        self.X = data.x.to(device)
        self.y = data.y.to(device)
        self.num_classes = dataset.num_classes
        self.num_nodes = data.num_nodes

        edge_index, values = get_laplacian(data.edge_index, num_nodes=data.num_nodes, normalization='sym')
        self.L = torch.sparse_coo_tensor(edge_index, values, (data.num_nodes, data.num_nodes), device=device)

        if atk == 'DICE':
            print(f"Applying DICE Attack (ratio: {ptb_rate})...")
            atk_edge_index = dice_attack(data.edge_index, num_nodes=data.num_nodes, y=data.y, ratio=ptb_rate)
            atk_edge_index, akt_values = get_laplacian(atk_edge_index, num_nodes=data.num_nodes, normalization='sym')
            self.atk_L = torch.sparse_coo_tensor(atk_edge_index, akt_values,
                                                  (data.num_nodes, data.num_nodes), device=device)
        elif atk == 'Metaattack':
            print(f"Applying Metaattack (ratio: {ptb_rate})...")
            atk_edge_index = metattack(data, ptb_rate)
            atk_edge_index, akt_values = get_laplacian(atk_edge_index, num_nodes=data.num_nodes, normalization='sym')
            self.atk_L = torch.sparse_coo_tensor(atk_edge_index, akt_values,
                                                  (data.num_nodes, data.num_nodes), device=device)
        elif atk == 'Noise':
            print(f"Applying Feature Noise Adding Attack (ratio: {ptb_rate})...")
            self.atk_X = node_perturbation(self.X, ratio=0.25, noise_scale=ptb_rate)

        self.train_mask = data.train_mask.to(device)
        self.val_mask = data.val_mask.to(device)
        self.test_mask = data.test_mask.to(device)

    def resplit(self, train_ratio=0.6):
        """Re-splits dataset splits for multi-run evaluation."""
        val_test_ratio = (1 - train_ratio) / 2
        transform = T.RandomNodeSplit(num_val=val_test_ratio, num_test=val_test_ratio)
        data = transform(self.data)
        self.train_mask = data.train_mask.to(self.device)
        self.val_mask = data.val_mask.to(self.device)
        self.test_mask = data.test_mask.to(self.device)
