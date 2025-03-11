import flwr as fl
from flwr.server.strategy import FedAvg
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="base", version_base=None)
def start_fl_server(cfg: DictConfig):
    """Start the Federated Learning server using Hydra config."""
    strategy = FedAvg(
        fraction_fit=cfg.server.fraction_fit,
        min_fit_clients=cfg.server.min_clients,
        min_available_clients=cfg.num_clients,
    )
    
    fl.server.start_server(
        server_address=cfg.server.address,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    start_fl_server()
