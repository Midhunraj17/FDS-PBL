import torch
from torch import nn
from .gcn import GCNLayer

class GCNGRU(nn.Module):
    """Lightweight missingness-aware spatial-temporal model.

    Input: [B,T,N,F], mask: [B,T,N,F], adjacency: [N,N].
    GCN layers operate on each time slice; a shared GRU then models
    temporal evolution independently for each sensor. Shared heads map
    each sensor's temporal state to forecast and imputation outputs.
    """
    def __init__(self, num_nodes, num_features, hidden_dim=16, horizon=3, dropout=0.1):
        super().__init__()
        self.n, self.f, self.h, self.p = num_nodes, num_features, hidden_dim, horizon
        self.input = nn.Linear(num_features * 2, hidden_dim)
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.forecast = nn.Linear(hidden_dim, horizon * num_features)
        self.impute = nn.Linear(hidden_dim, num_features)

    def forward(self, x, mask, adjacency):
        z = torch.cat([x, mask], dim=-1)
        z = torch.relu(self.input(z))
        z = torch.relu(self.gcn1(z, adjacency))
        z = torch.relu(self.gcn2(z, adjacency))
        b, t, n, h = z.shape
        # One shared temporal GRU is applied to every sensor sequence.
        node_seq = z.permute(0, 2, 1, 3).reshape(b * n, t, h)
        temporal, _ = self.gru(node_seq)
        last = self.dropout(temporal[:, -1])
        delta = self.forecast(last).reshape(b, n, self.p, self.f).permute(0, 2, 1, 3)
        observed_count = mask.sum(dim=1).clamp_min(1.0)
        recent_baseline = (x * mask).sum(dim=1) / observed_count
        forecast = recent_baseline.unsqueeze(1) + delta
        imputation_last = self.impute(last).reshape(b, n, self.f)
        imputation = imputation_last.unsqueeze(1).expand(-1, t, -1, -1)
        return {'forecast': forecast, 'imputation': imputation}
