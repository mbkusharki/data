import os
import hydra
from omegaconf import DictConfig, OmegaConf
import ray
import subprocess
from server import start_fl_server
from client import start_fl_client

# ✅ Critical fixes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP workaround
os.environ["RAY_DISABLE_WINDOWS_SYMBOLIZER"] = "1"  # Fix Ray symbolizer issue
os.environ["RAY_BACKEND_LOG_LEVEL"] = "ERROR"  # Reduce Ray logs

# ✅ Fix Ray Worker Crashes & Reduce Logs
os.environ["RAY_worker_register_timeout_seconds"] = "60"
os.environ["RAY_DISABLE_LOG_MONITOR"] = "1"

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """Run the Federated Learning system using Hydra & Ray."""
    print("🔧 Loaded Configuration:\n", OmegaConf.to_yaml(cfg))

    # ✅ Initialize Ray with Error Handling
    for attempt in range(3):  # Retry up to 3 times
        try:
            ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=cfg.num_clients)
            print("✅ Ray Successfully Initialized")
            break
        except Exception as e:
            print(f"❌ Ray Initialization Failed (Attempt {attempt + 1}): {e}")
            if attempt == 2:
                return  # Exit after 3 failed attempts

    # ✅ Start the FL Server
    server_proc = subprocess.Popen(["python", "server.py"])

    @ray.remote
    def run_client(farm_id):
        """Ray-Parallelized FL Client Execution"""
        start_fl_client(farm_id, cfg)

    # ✅ Start clients in parallel
    client_tasks = [run_client.remote(farm) for farm in ["Farm_1", "Farm_2", "Farm_3"]]
    ray.get(client_tasks)  # ✅ Wait for all clients to finish

    server_proc.terminate()
    print("✅ Federated Learning Completed!")

if __name__ == "__main__":
    main()