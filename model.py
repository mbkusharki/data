import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class FeGAN(nn.Module):
    def __init__(self, in_channels=128, hidden_channels=64, out_channels=4, heads=8, dropout=0.2):
        """
        Federated Graph Attention Network (FeGAN) for disease classification.
        Args:
            in_channels (int): Number of input features per node.
            hidden_channels (int): Number of hidden layer features.
            out_channels (int): Number of output classes (disease categories).
            heads (int): Number of attention heads in GAT layers.
            dropout (float): Dropout rate for regularization.
        """
        super(FeGAN, self).__init__()

        # First GAT layer (multi-head attention)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        
        # Second GAT layer (aggregates multi-head outputs)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        """
        Forward pass of FeGAN model.
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge connections between nodes.
        Returns:
            Tensor: Log-softmax output for classification.
        """
        x = F.elu(self.gat1(x, edge_index))  # Apply first GAT layer + activation
        x = self.gat2(x, edge_index)  # Apply second GAT layer
        return F.log_softmax(x, dim=1)  # Output probabilities (log-softmax)
