import torch
import numpy as np
from src import DenseNet3D, FeatureExtractor  

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize 3D model
model = DenseNet3D().to(device)
print("3D DenseNet initialized.")

# 2. Create dummy 3D inputs simulating Tibia, Femur, Background
# Shape: (B=1, C=3, D=16, H=112, W=112)
regions_dict = {
    'tibia': torch.randn(1, 3, 16, 112, 112).to(device),
    'femur': torch.randn(1, 3, 16, 112, 112).to(device),
    'background': torch.randn(1, 3, 16, 112, 112).to(device),
}

# 3. Initialize FeatureExtractor
extractor = FeatureExtractor(model)

# 4. Extract features
features = extractor.extract_features(regions_dict)

# 5. Compute cosine similarities
similarities_dict, results_df = extractor.compute_cosine_similarities(features)

# 6. Save CSV
csv_path = "similarity_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"Saved similarity results to: {csv_path}")
