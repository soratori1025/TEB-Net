import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.nn import NNConv, GCNConv, GATConv, global_add_pool
from torch_geometric.loader import DataLoader

class GNF_MPNN(nn.Module):
    """
    Message Passing Neural Network (GNF-MPNN):
    - Radius (MP Layers): 3
    - Embedding Dimension: 256
    - Edge Feature Dimension: 10
    - Readout: Global sum pooling -> Dense hidden 256 (ReLU, no dropout, no bias) -> 1
    """
    def __init__(self, node_in_dim=26, edge_in_dim=10, emb_dim=256, radius=3):
        super(GNF_MPNN, self).__init__()
        self.radius = radius
        self.node_emb = nn.Linear(node_in_dim, emb_dim)
        
        # Message passing layers using NNConv (edge condition network)
        self.convs = nn.ModuleList()
        for _ in range(radius):
            edge_nn = nn.Sequential(
                nn.Linear(edge_in_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim * emb_dim)
            )
            self.convs.append(NNConv(emb_dim, emb_dim, edge_nn, aggr='add'))
            
        # Dense readout head: 1 hidden layer of 256 neurons, ReLU, no dropout, no bias
        self.readout_hidden = nn.Linear(emb_dim, emb_dim, bias=False)
        self.act = nn.ReLU()
        self.out = nn.Linear(emb_dim, 1, bias=False)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Initial node embedding
        h = self.act(self.node_emb(x))
        
        # Message passing
        for conv in self.convs:
            h = self.act(conv(h, edge_index, edge_attr))
            
        # Global pooling (sum) to get GNF fingerprint
        gnf = global_add_pool(h, batch)
        
        # Dense prediction head
        h_out = self.act(self.readout_hidden(gnf))
        out = self.out(h_out)
        return out

class GNF_GCN(nn.Module):
    """
    Graph Convolutional Network (GNF-GCN)
    """
    def __init__(self, node_in_dim=26, emb_dim=256, radius=3):
        super(GNF_GCN, self).__init__()
        self.node_emb = nn.Linear(node_in_dim, emb_dim)
        self.convs = nn.ModuleList([GCNConv(emb_dim, emb_dim) for _ in range(radius)])
        self.act = nn.ReLU()
        self.readout_hidden = nn.Linear(emb_dim, emb_dim, bias=False)
        self.out = nn.Linear(emb_dim, 1, bias=False)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.act(self.node_emb(x))
        for conv in self.convs:
            h = self.act(conv(h, edge_index))
        gnf = global_add_pool(h, batch)
        h_out = self.act(self.readout_hidden(gnf))
        return self.out(h_out)

class GNF_GAT(nn.Module):
    """
    Graph Attention Network (GNF-GAT)
    """
    def __init__(self, node_in_dim=26, emb_dim=256, radius=3, heads=4):
        super(GNF_GAT, self).__init__()
        self.node_emb = nn.Linear(node_in_dim, emb_dim)
        self.convs = nn.ModuleList()
        for i in range(radius):
            in_c = emb_dim if i == 0 else emb_dim
            self.convs.append(GATConv(in_c, emb_dim // heads, heads=heads))
        self.act = nn.ReLU()
        self.readout_hidden = nn.Linear(emb_dim, emb_dim, bias=False)
        self.out = nn.Linear(emb_dim, 1, bias=False)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.act(self.node_emb(x))
        for conv in self.convs:
            h = self.act(conv(h, edge_index))
        gnf = global_add_pool(h, batch)
        h_out = self.act(self.readout_hidden(gnf))
        return self.out(h_out)

class GNNTrainer:
    """
    Trainer for PyTorch Geometric GNN models.
    """
    def __init__(self, model_type='GNF-MPNN', lr=0.001, epochs=400, batch_size=128, device=None):
        self.model_type = model_type
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_type == 'GNF-MPNN':
            self.model = GNF_MPNN().to(self.device)
        elif model_type == 'GNF-GCN':
            self.model = GNF_GCN().to(self.device)
        elif model_type == 'GNF-GAT':
            self.model = GNF_GAT().to(self.device)
        else:
            raise ValueError(f"Unknown GNN model type: {model_type}")
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def fit(self, train_graphs):
        self.model.train()
        loader = DataLoader(train_graphs, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            for batch in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(batch)
                loss = self.criterion(out, batch.y.view(-1, 1))
                loss.backward()
                self.optimizer.step()

    def predict(self, test_graphs):
        self.model.eval()
        loader = DataLoader(test_graphs, batch_size=self.batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                preds.append(out.cpu().numpy().ravel())
        return np.concatenate(preds)
