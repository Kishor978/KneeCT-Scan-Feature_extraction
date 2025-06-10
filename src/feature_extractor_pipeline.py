# utils/similarity.py

import torch
import pandas as pd
from src import DenseNet3D, FeatureExtractor

def densenet_pipeline(csv_path="similarity_results.csv", regions_dict=None, device=None):
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize 3D model
    model = DenseNet3D().to(device)
    print("3D DenseNet initialized.")

    # 3. Initialize FeatureExtractor
    extractor = FeatureExtractor(model)

    # 4. Extract features
    features = extractor.extract_features(regions_dict)

    # 5. Compute cosine similarities
    similarities_dict, results_df = extractor.compute_cosine_similarities(features)

    # 6. Save CSV
    results_df.to_csv(csv_path, index=False)
    print(f"Saved similarity results to: {csv_path}")

    return similarities_dict, results_df
