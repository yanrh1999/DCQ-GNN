import torch
import torch.nn as nn
from torch.sparse import mm
from torch.nn import functional as F


class Classifier(nn.Module):
    """Two-layer MLP classifier head with dropout."""
    def __init__(self, in_features: int, out_features: int):
        super(Classifier, self).__init__()
        self.mlps = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features // 2, out_features)
        )
        nn.init.kaiming_uniform_(self.mlps[0].weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.mlps[-1].weight)

    def forward(self, emb):
        return self.mlps(emb)


class QuadraticFilterLayer(nn.Module):
    """
    A single DCQ-GNN layer implementing the adaptive quadratic filter bank
    with node-adaptive gated fusion.

    Four parallel channels:
      - Low-pass:  homophilic smoothing (convex / concave)
      - High-pass: heterophilic disassortativity (convex / concave)
      - Mid-pass:  structural perturbation shield
      - All-pass:  structure-agnostic identity anchor

    Learnable parameters:
      - lambda_t (3,): cutoff proxy per channel, clamped to [0, 2]
      - beta (4,):    scaling factor per channel, clamped to [0.01, 1]
    """
    def __init__(self, hidden_features: int, dr: float, opt: list = ['cvx', 'cvx']):
        super(QuadraticFilterLayer, self).__init__()
        self.lambda_t = nn.Parameter(torch.tensor([2, 2, 2], dtype=torch.float), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor([1, 1, 1, 1], dtype=torch.float), requires_grad=True)
        self.opt = opt

        self.gate = nn.ModuleList()
        for i in range(4):
            self.gate.append(nn.Sequential(
                nn.Dropout(p=dr),
                nn.Linear(hidden_features, hidden_features, bias=False),
                nn.Hardsigmoid()
            ))
            nn.init.xavier_uniform_(self.gate[-1][1].weight)

        self.weight = nn.Sequential(
            nn.Dropout(p=dr),
            nn.BatchNorm1d(hidden_features),
            nn.Linear(hidden_features, hidden_features)
        )
        nn.init.kaiming_uniform_(self.weight[-1].weight, a=0, mode='fan_in', nonlinearity='relu')
        self.register_forward_pre_hook(self._clamp_hook)

    @staticmethod
    def _clamp_hook(module, input):
        with torch.no_grad():
            module.lambda_t.data.clamp_(0, 2)
            module.beta.data.clamp_(min=0.01, max=1)

    def low_pass(self, emb, L, I):
        """Low-pass channel: H = (β/λ²) * (L - λI)² * X (convex) or its negation (concave)."""
        out = emb
        if self.opt == 'cvx':
            out = mm(L - self.lambda_t[0] * I, out)
            out = mm(L - self.lambda_t[0] * I, out)
            out = (self.beta[0] / self.lambda_t[0] ** 2) * out
        else:
            out = mm(L - self.lambda_t[0] * I, out)
            out = mm(L - self.lambda_t[0] * I, out)
            out = -(self.beta[0] / self.lambda_t[0] ** 2) * out
        return out

    def high_pass(self, emb, L, I):
        """High-pass channel: H = (β/λ²) * L² * X (convex) or its negation (concave)."""
        out = emb
        if self.opt == 'cvx':
            out = mm(L, out)
            out = mm(L, out)
            out = (self.beta[1] / self.lambda_t[1] ** 2) * out
        else:
            out = mm(L - 2 * self.lambda_t[1] * I, out)
            out = mm(L, out)
            out = -(self.beta[1] / self.lambda_t[1] ** 2) * out
        return out

    def mid_pass(self, emb, L, I):
        """Mid-pass channel: H = -(4β/λ²) * L * (L - λI) * X."""
        out = emb
        out = mm(L - self.lambda_t[2] * I, out)
        out = mm(L, out)
        out = -(4 * self.beta[2] / self.lambda_t[2] ** 2) * out
        return out

    def all_pass(self, emb, I):
        """All-pass channel: H = β * X (identity anchor)."""
        out = emb
        out = mm(self.beta[3] * I, out)
        return out

    def forward(self, emb, L, I, gate_save=False, current_layer=0):
        """Node-adaptive gated fusion of the four filter channels."""
        out = None
        filters = ['low_pass', 'high_pass', 'mid_pass', 'all_pass']
        tmp = [self.low_pass(emb, L, I), self.high_pass(emb, L, I),
               self.mid_pass(emb, L, I), self.all_pass(emb, I)]
        for i, tmp_emb in enumerate(tmp):
            gate = self.gate[i](emb)
            if gate_save:
                torch.save(gate.cpu(), f'./gate_weights/layer{current_layer}_{filters[i]}_gate.pt')
            if out is None:
                out = tmp_emb * gate
            else:
                out = out + tmp_emb * gate
        out = self.weight(out)
        return out


class DCQ_GNN(nn.Module):
    """
    DCQ-GNN: Dual-Channel Quadratic Graph Neural Network.

    Stacks multiple QuadraticFilterLayers. Each layer applies a bank of
    second-order (quadratic) spectral filters and fuses them via learned
    node-wise gates.  Supports optional multi-scale skip connections.

    Architecture:
      1. Linear projection to hidden space
      2. Stacked QuadraticFilterLayers
      3. Classifier head (MLP)
    """
    def __init__(self, num_layer: int, in_features: int, hidden_features: int, out_features: int,
                 dr, opt: list, multi_scale: bool = False):
        super(DCQ_GNN, self).__init__()
        self.num_layer = num_layer
        self.layers = nn.ModuleList()
        self.multi_scale = multi_scale

        self.linear = nn.Sequential(nn.Dropout(dr), nn.Linear(in_features, hidden_features), nn.ReLU())
        for layer in range(num_layer):
            self.layers.append(QuadraticFilterLayer(hidden_features, dr=dr,
                                                     opt=[opt[0][layer], opt[1][layer]]))
        if multi_scale:
            self.classifier = Classifier(in_features=((num_layer + 1) * hidden_features),
                                         out_features=out_features)
        else:
            self.classifier = Classifier(in_features=hidden_features, out_features=out_features)

    def forward(self, X, L, I, gate_save=False):
        out = self.linear(X)
        if self.multi_scale:
            multi_scale_out = out
            for layer in range(self.num_layer):
                out = self.layers[layer](out, L, I, gate_save=gate_save, current_layer=layer)
                multi_scale_out = torch.cat([multi_scale_out, out], dim=1)
            out = multi_scale_out
        else:
            for layer in range(self.num_layer):
                out = self.layers[layer](out, L, I, gate_save=gate_save, current_layer=layer)
        out = self.classifier(out)
        return out
