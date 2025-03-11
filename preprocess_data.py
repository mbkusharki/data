import hydra
from omegaconf import DictConfig
from dataset import save_graph_data

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """Convert images to graph and save as .pt files for each farm."""
    if not hasattr(cfg, 'privacy') or not hasattr(cfg.privacy, 'noise_level'):
        raise ValueError("Configuration is missing 'privacy.noise_level'.")
    if not hasattr(cfg, 'dataset') or not hasattr(cfg.dataset, 'image_size') or not hasattr(cfg.dataset, 'feature_dim'):
        raise ValueError("Configuration is missing 'dataset.image_size' or 'dataset.feature_dim'.")

    for farm in ["Farm_1", "Farm_2", "Farm_3"]:  # Processing each farm

        print(f"📌 Processing {farm}...")
        save_graph_data(farm, cfg)
    print("✅ Graph conversion completed!")

if __name__ == "__main__":
    main()
