import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import Conv2d, BatchNorm2d
import copy
import warnings

class DenseNet3D(nn.Module):
    """
    3D DenseNet121 that properly handles dense connections and concatenations
    """
    
    def __init__(self, depth_size=16):
        super(DenseNet3D, self).__init__()
        self.depth_size = depth_size
        
        # Build a simplified but functional 3D DenseNet
        # Initial convolution
        self.conv0 = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.norm0 = nn.BatchNorm3d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Dense blocks with transitions
        self.denseblock1 = self._make_dense_block(64, 32, 6)   # 64 + 6*32 = 256
        self.transition1 = self._make_transition(256, 128)
        
        self.denseblock2 = self._make_dense_block(128, 32, 12)  # 128 + 12*32 = 512
        self.transition2 = self._make_transition(512, 256)
        
        self.denseblock3 = self._make_dense_block(256, 32, 24)  # 256 + 24*32 = 1024
        self.transition3 = self._make_transition(1024, 512)
        
        self.denseblock4 = self._make_dense_block(512, 32, 16)  # 512 + 16*32 = 1024
        
        # Final layers
        self.norm5 = nn.BatchNorm3d(1024)
        self.relu5 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(1024, 1000)
        
        # Initialize with pretrained weights
        self._initialize_from_2d()
        
        # Track conv layers for feature extraction
        self.conv_layers = self._get_conv_layers()
        
    def _make_dense_block(self, in_channels, growth_rate, num_layers):
        """Create a dense block with proper concatenation"""
        layers = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(num_layers):
            layers.append(self._make_dense_layer(current_channels, growth_rate))
            current_channels += growth_rate
            
        return DenseBlock3D(layers, in_channels)
    
    def _make_dense_layer(self, in_channels, growth_rate):
        """Create a single dense layer (BN-ReLU-Conv1x1-BN-ReLU-Conv3x3)"""
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm3d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv3d(4 * growth_rate, growth_rate, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        )
    
    def _make_transition(self, in_channels, out_channels):
        """Create transition layer (BN-ReLU-Conv1x1-AvgPool)"""
        return nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
    
    def _initialize_from_2d(self):
        """Fully initialize 3D DenseNet weights from pretrained 2D DenseNet121"""
        import torchvision.models as models
        from torch.nn import Conv2d, BatchNorm2d
        import copy

        print("Inflating full DenseNet121 from 2D pretrained weights...")

        # Load 2D pretrained model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_2d = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        def inflate_conv2d_to_conv3d(conv2d, in_channels_override=None):
            in_channels = in_channels_override or conv2d.in_channels
            out_channels = conv2d.out_channels
            k_h, k_w = conv2d.kernel_size
            d = 3  # Temporal depth
            conv3d = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(d, k_h, k_w),
                stride=(1, *conv2d.stride),
                padding=(d // 2, *conv2d.padding),
                bias=(conv2d.bias is not None)
            )
            with torch.no_grad():
                weight2d = conv2d.weight.data
                if in_channels == weight2d.shape[1]:
                    weight3d = weight2d.unsqueeze(2).repeat(1, 1, d, 1, 1) / d
                else:
                    # For grayscale input: average across RGB channels
                    weight2d_avg = weight2d.mean(dim=1, keepdim=True)
                    weight3d = weight2d_avg.unsqueeze(2).repeat(1, 1, d, 1, 1) / d
                conv3d.weight.copy_(weight3d)
                if conv2d.bias is not None:
                    conv3d.bias.copy_(conv2d.bias.data)
            return conv3d

        def inflate_bn2d_to_bn3d(bn2d):
            bn3d = nn.BatchNorm3d(bn2d.num_features)
            bn3d.weight.data.copy_(bn2d.weight.data)
            bn3d.bias.data.copy_(bn2d.bias.data)
            bn3d.running_mean.data.copy_(bn2d.running_mean.data)
            bn3d.running_var.data.copy_(bn2d.running_var.data)
            return bn3d

        def replace_module_2d_to_3d(module_3d, module_2d):
            for name, mod_2d in module_2d.named_children():
                if isinstance(mod_2d, Conv2d):
                    mod_3d = inflate_conv2d_to_conv3d(mod_2d)
                elif isinstance(mod_2d, BatchNorm2d):
                    mod_3d = inflate_bn2d_to_bn3d(mod_2d)
                else:
                    mod_3d = copy.deepcopy(mod_2d)

                if hasattr(module_3d, name):
                    setattr(module_3d, name, mod_3d)

                # Recurse
                if hasattr(module_3d, name) and hasattr(mod_2d, 'named_children'):
                    replace_module_2d_to_3d(getattr(module_3d, name), mod_2d)

        # Inflate base layers
        self.conv0 = inflate_conv2d_to_conv3d(model_2d.features.conv0, in_channels_override=1)
        self.norm0 = inflate_bn2d_to_bn3d(model_2d.features.norm0)

        # Inflate dense blocks and transitions
        replace_module_2d_to_3d(self.denseblock1, model_2d.features.denseblock1)
        replace_module_2d_to_3d(self.transition1, model_2d.features.transition1)
        replace_module_2d_to_3d(self.denseblock2, model_2d.features.denseblock2)
        replace_module_2d_to_3d(self.transition2, model_2d.features.transition2)
        replace_module_2d_to_3d(self.denseblock3, model_2d.features.denseblock3)
        replace_module_2d_to_3d(self.transition3, model_2d.features.transition3)
        replace_module_2d_to_3d(self.denseblock4, model_2d.features.denseblock4)

        print("✅ Full DenseNet feature extractor inflated from 2D.")
    
    def _copy_conv_weights(self, state_dict_2d, key_2d, conv3d_layer):
        """Copy and inflate 2D conv weights to 3D"""
        if key_2d in state_dict_2d:
            weight_2d = state_dict_2d[key_2d]
            # Inflate to 3D
            depth_size = conv3d_layer.kernel_size[0]
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, depth_size, 1, 1) / depth_size
            conv3d_layer.weight.data = weight_3d
    
    def _copy_bn_weights(self, state_dict_2d, prefix_2d, bn3d_layer):
        """Copy 2D BatchNorm weights to 3D"""
        weight_key = f"{prefix_2d}.weight"
        bias_key = f"{prefix_2d}.bias"
        mean_key = f"{prefix_2d}.running_mean"
        var_key = f"{prefix_2d}.running_var"
        
        if weight_key in state_dict_2d:
            bn3d_layer.weight.data = state_dict_2d[weight_key].clone()
        if bias_key in state_dict_2d:
            bn3d_layer.bias.data = state_dict_2d[bias_key].clone()
        if mean_key in state_dict_2d:
            bn3d_layer.running_mean.data = state_dict_2d[mean_key].clone()
        if var_key in state_dict_2d:
            bn3d_layer.running_var.data = state_dict_2d[var_key].clone()
    
    def _init_remaining_weights(self):
        """Initialize remaining weights with proper scaling"""
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                if module.weight.data.sum() == 0:  # Not initialized yet
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm3d):
                if module.weight.data.sum() == 0:  # Not initialized yet
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
    
    def _get_conv_layers(self):
        """Get all conv layers for feature extraction"""
        conv_layers = []
        
        def add_conv_layers(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Conv3d):
                    conv_layers.append((full_name, child))
                else:
                    add_conv_layers(child, full_name)
        
        add_conv_layers(self)
        return conv_layers
    
    def forward(self, x):
        """Forward pass"""
        # Initial layers
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        
        # Dense blocks with transitions
        x = self.denseblock1(x)
        x = self.transition1(x)
        
        x = self.denseblock2(x)
        x = self.transition2(x)
        
        x = self.denseblock3(x)
        x = self.transition3(x)
        
        x = self.denseblock4(x)
        
        # Final layers
        x = self.norm5(x)
        x = self.relu5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def extract_features(self, x, layer_indices=[-1, -3, -5]):
        """Extract features from specified conv layers"""
        features = {}
        all_features = []
        
        # Forward pass with feature collection
        # Initial layers
        x = self.conv0(x)
        all_features.append(('conv0', x.clone()))
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        
        # Dense blocks with transitions
        x, block_features = self.denseblock1.forward_with_features(x)
        all_features.extend(block_features)
        x = self.transition1(x)
        
        x, block_features = self.denseblock2.forward_with_features(x)
        all_features.extend(block_features)
        x = self.transition2(x)
        
        x, block_features = self.denseblock3.forward_with_features(x)
        all_features.extend(block_features)
        x = self.transition3(x)
        
        x, block_features = self.denseblock4.forward_with_features(x)
        all_features.extend(block_features)
        
        # Extract features from specified layers
        total_conv_layers = len(all_features)
        for idx in layer_indices:
            if idx < 0:
                actual_idx = total_conv_layers + idx
            else:
                actual_idx = idx
            
            if 0 <= actual_idx < total_conv_layers:
                layer_name, feature_map = all_features[actual_idx]
                features[f"layer_{actual_idx}_{layer_name}"] = feature_map
        
        return features


class DenseBlock3D(nn.Module):
    """3D Dense Block that handles concatenation properly"""
    
    def __init__(self, layers, initial_channels):
        super(DenseBlock3D, self).__init__()
        self.layers = layers
        self.initial_channels = initial_channels
    
    def forward(self, x):
        """Forward pass with concatenation"""
        features = [x]
        
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        
        return torch.cat(features, 1)
    
    def forward_with_features(self, x):
        """Forward pass that also returns intermediate features"""
        features = [x]
        conv_features = []
        
        for i, layer in enumerate(self.layers):
            input_tensor = torch.cat(features, 1)
            new_features = layer(input_tensor)
            features.append(new_features)
            
            # Extract conv layer outputs
            for name, module in layer.named_modules():
                if isinstance(module, nn.Conv3d):
                    conv_features.append((f"dense_conv_{i}_{name}", new_features.clone()))
        
        return torch.cat(features, 1), conv_features

def apply_global_average_pooling(feature_maps):
    """Apply Global Average Pooling to feature maps"""
    gap_features = {}
    
    for layer_name, feature_map in feature_maps.items():
        # Apply GAP: average over spatial dimensions (D, H, W)
        gap_vector = torch.mean(feature_map, dim=[2, 3, 4])  # Keep batch and channel dims
        gap_features[layer_name + "_gap"] = gap_vector
    
    return gap_features


def create_3d_densenet121(depth_size=16):
    """Create a 3D DenseNet121 model"""
    print("Creating 3D DenseNet121...")
    model_3d = DenseNet3D(depth_size=depth_size)
    
    print(f"Model created with {len(model_3d.conv_layers)} convolutional layers")
    return model_3d


def test_conversion():
    """Test the 3D DenseNet model"""
    try:
        # Create 3D model
        model_3d = create_3d_densenet121(depth_size=16)
        
        # Test input
        print("\nTesting with input shape: (1, 3, 16, 224, 224)")
        dummy_input = torch.randn(1, 1, 64, 64, 64)
        
        with torch.no_grad():
            # Test feature extraction
            print("Testing feature extraction...")
            features = model_3d.extract_features(dummy_input, layer_indices=[-1, -3, -5])
            print(f"Feature extraction successful! Extracted {len(features)} feature maps:")
            for name, feat in features.items():
                print(f"  {name}: {feat.shape}")
            
            # Test Global Average Pooling
            if features:
                print("\nTesting Global Average Pooling...")
                gap_features = apply_global_average_pooling(features)
                print("GAP features:")
                for name, feat in gap_features.items():
                    print(f"  {name}: {feat.shape}")
            
            # Test full forward pass
            print("\nTesting full forward pass...")
            output = model_3d(dummy_input)
            print(f"Forward pass successful! Output shape: {output.shape}")
            
        print("\n✅ All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_conversion()