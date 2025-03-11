import os
import torch
import numpy as np
import cv2
from torch_geometric.data import Data
from skimage.feature import canny
from scipy.spatial import KDTree

def add_noise(image, noise_level=0.1):
    """Apply Gaussian noise for privacy."""
    noise = np.random.normal(0, noise_level * 255, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 255)

def image_to_graph(image_path, noise_level, image_size, feature_dim):
    """Convert an image into a graph structure for GAT with memory optimization."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (image_size, image_size))
    img = add_noise(img, noise_level)

    edges = canny(img / 255.0)
    edge_positions = np.argwhere(edges > 0)

    if len(edge_positions) < 2:
        return None  # Skip if no features

    # Create nearest-neighbor edges
    tree = KDTree(edge_positions)
    edge_index = []
    for i, pos in enumerate(edge_positions):
        _, neighbors = tree.query(pos, k=3)
        for neighbor in neighbors[1:]:
            edge_index.append([i, neighbor])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Reduce memory by limiting feature dimension expansion
    try:
        x = torch.tensor(img.flatten(), dtype=torch.float).view(-1, 1)
        x = torch.cat([x, torch.randn(x.shape[0], min(64, feature_dim - 1))], dim=1)  # Reduce to 64 features max


    except RuntimeError:
        print(f"❌ Memory Error on {image_path}, skipping...")
        return None

    return Data(x=x, edge_index=edge_index)

def save_graph_data(farm_id, cfg):
    """Convert images in batches and save all graphs of a farm into a single .pt file."""
    farm_path = os.path.join("Data", farm_id)
    graph_dir = "graph_data"  # New single folder for all farms' .pt files
    os.makedirs(graph_dir, exist_ok=True)

    farm_graph_path = os.path.join(graph_dir, f"{farm_id}.pt")

    total_images = 0  # Count all images before conversion
    max_images_per_disease = 350  # Limit to 350 images per disease
    total_images_per_farm = 1000  # Limit to 1000 images per farm

    disease_image_counts = {}  # Dictionary to store image counts per disease

    # First, count all images per disease BEFORE converting
    for crop_folder in os.listdir(farm_path):
        crop_path = os.path.join(farm_path, crop_folder)
        if not os.path.isdir(crop_path):
            continue  # Ensure it's a directory (Crop folder)

        for disease_folder in os.listdir(crop_path):
            disease_path = os.path.join(crop_path, disease_folder)
            if not os.path.isdir(disease_path):
                continue  # Ensure it's a directory (Disease folder)

            num_images = len([f for f in os.listdir(disease_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if disease_folder not in disease_image_counts:
                disease_image_counts[disease_folder] = 0  # Initialize count for this disease

            limited_images = min(num_images, max_images_per_disease)
            total_images += limited_images
            disease_image_counts[disease_folder] += limited_images  # Store in dict


    # Print the total image count for each disease in the farm
    for disease, count in disease_image_counts.items():
        print(f"📊 {farm_id} - {disease}: {count} images")

    
    print(f"📢 Total images in {farm_id}: {total_images} (using only 350 images per disease for conversion)\n")





    batch_size = 100  # Process images in batches of 100 to reduce memory use

    processed_count = 0
    farm_graphs = []  # Store all graphs of the farm in one list

    # Now, process images in batches
    for crop_folder in os.listdir(farm_path):
        crop_path = os.path.join(farm_path, crop_folder)
        if not os.path.isdir(crop_path):
            continue  # Ensure it's a directory (Crop folder)

        for disease_folder in os.listdir(crop_path):
            disease_path = os.path.join(crop_path, disease_folder)
            if not os.path.isdir(disease_path):
                continue  # Ensure it's a directory (Disease folder)

            disease_graphs = []  # Temporary list for batching

            img_files = [f for f in os.listdir(disease_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in img_files[:max_images_per_disease]:  # Limit to max_images_per_disease

                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):  
                    img_path = os.path.join(disease_path, img_file)
                    graph = image_to_graph(img_path, cfg.privacy.noise_level, cfg.dataset.image_size, cfg.dataset.feature_dim)

                    if graph:
                        disease_graphs.append(graph)
                        disease_image_counts[disease_folder] += 1  # Increment count for this disease
                        processed_count += 1

                    # Process in batches to prevent memory overload
                    if len(disease_graphs) >= batch_size or processed_count >= total_images_per_farm:
                        farm_graphs.extend(disease_graphs)  # Add to farm-wide list
                        disease_graphs = []  # Reset batch list

            # Add remaining graphs, but only if we haven't reached the limit
            if disease_graphs and processed_count <= total_images_per_farm:
                farm_graphs.extend(disease_graphs)

    # Save all graphs for the farm in one .pt file
    if farm_graphs:
        farm_graph_path = os.path.join(graph_dir, f"{farm_id}.pt")
        torch.save(farm_graphs, farm_graph_path)
        print(f"✅ Saved {len(farm_graphs)} graphs for {farm_id} into {farm_id}.pt")

    print(f"✅ Done processing {farm_id}")

def load_graph_data(farm_id):
    """Load processed graph data for a farm."""
    graph_dir = "graph_data"
    graph_path = os.path.join(graph_dir, f"{farm_id}.pt")

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"❌ Graph file missing: {graph_path}")

    dataset = torch.load(graph_path)
    print(f"✅ Loaded {len(dataset)} graph samples from {farm_id}")
    return dataset
