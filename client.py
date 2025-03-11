import flwr as fl
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import FeGAN
from dataset import load_graph_data

class FeGANClient(fl.client.NumPyClient):
    def __init__(self, farm_id, cfg):
        self.farm_id = farm_id
        self.model = FeGAN(in_channels=cfg.dataset.feature_dim, hidden_channels=64)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        # ✅ Load preprocessed graph dataset from `graph_data/`
        print(f"📥 Loading graph data for {farm_id}...")
        self.dataset = load_graph_data(farm_id)
        print(f"✅ {len(self.dataset)} graph samples loaded for {farm_id}!")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for _ in range(3):  # ✅ 3 local epochs per farm
            for data in self.dataset:
                self.optimizer.zero_grad()
                out = self.model(data.x, data.edge_index)
                loss = F.nll_loss(out, torch.randint(0, 4, (data.x.size(0),)))  # Dummy labels for testing
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config), len(self.dataset), {}

def start_fl_client(farm_id, cfg):
    """Start FL Client for a given farm."""
    client = FeGANClient(farm_id, cfg)
    fl.client.start_numpy_client(cfg.server.address, client=client)
