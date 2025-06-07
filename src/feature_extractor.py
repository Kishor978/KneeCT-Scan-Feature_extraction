import torch    
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class FeatureExtractor:
    def __init__(self, model_3d):
        self.model = model_3d
        self.model.eval()
        
        self.target_indices = [-1, -3, -5]  # layer indices for extract_features()
        self.layer_labels = [f"layer_{abs(i)}" for i in self.target_indices]  # e.g., layer_1, layer_3, ...

    def extract_features(self, regions_dict):
        features_dict = {}
        with torch.no_grad():
            for region_name, region_tensor in regions_dict.items():
                print(f"Extracting features from {region_name}...")
                layer_features = self.model.extract_features(region_tensor, layer_indices=self.target_indices)
                
                region_features = {}
                for layer_name, fmap in layer_features.items():
                    pooled = torch.mean(fmap, dim=[2, 3, 4]).squeeze(0)
                    region_features[layer_name] = pooled.cpu().numpy()
                features_dict[region_name] = region_features
        return features_dict

    def compute_cosine_similarities(self, features_dict):
        
        region_pairs = [('tibia', 'femur'), ('tibia', 'background'), ('femur', 'background')]
        similarities_dict = {}
        row_data = {'image_id': 'sample_1'}

        for layer_name in features_dict['tibia'].keys():  # same keys across regions
            for r1, r2 in region_pairs:
                f1 = features_dict[r1][layer_name].reshape(1, -1)
                f2 = features_dict[r2][layer_name].reshape(1, -1)
                sim = cosine_similarity(f1, f2)[0, 0]
                col_name = f"{layer_name}_{r1}_vs_{r2}"
                row_data[col_name] = sim

        df = pd.DataFrame([row_data])
        return similarities_dict, df
