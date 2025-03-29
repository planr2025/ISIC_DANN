import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm


# def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
#                     device: torch.device, max_num_features=None) -> torch.Tensor:
#     """
#     Fetch data from `data_loader`, and then use `feature_extractor` to collect features

#     Args:
#         data_loader (torch.utils.data.DataLoader): Data loader.
#         feature_extractor (torch.nn.Module): A feature extractor.
#         device (torch.device)
#         max_num_features (int): The max number of features to return

#     Returns:
#         Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
#     """
#     feature_extractor.eval()
#     all_features = []
#     with torch.no_grad():
#         for i, data in enumerate(tqdm.tqdm(data_loader)):
#             if max_num_features is not None and i >= max_num_features:
#                 break
#             inputs = data[0].to(device)  # (B, C, H, W)
#             # print(f"Original Input Shape: {inputs.shape}")  

#             # Remove batch dimension if B=1
#             if inputs.dim() == 4 and inputs.shape[0] == 1:
#                 inputs = inputs.squeeze(0)  # (C, H, W)

#             # print(f"Modified Input Shape: {inputs.shape}")

#             feature = feature_extractor(inputs).cpu()
#             all_features.append(feature)
#     return torch.cat(all_features, dim=0)

# def collect_feature(dataloader, feature_extractor, device):
#     features = []
#     with torch.no_grad():
#         for i, data in enumerate(tqdm.tqdm(dataloader, desc="Extracting Features", leave=True)):
#             inputs = data[0].to(device)  # (B, 3, 224, 224)
            
#             # Extract features step by step
#             # feature = feature_extractor[1](inputs)  # Pooling layer
#             # feature = feature.unsqueeze(-1).unsqueeze(-1)  # Ensure correct shape
#             # feature = feature_extractor[2](feature)  # Bottleneck layer

#             # print(f"Input shape: {inputs.shape}")
#             feature = feature_extractor[1](inputs)
#             # print(f"After pooling: {feature.shape}")
#             feature = feature.unsqueeze(-1).unsqueeze(-1)
#             # print(f"After unsqueeze: {feature.shape}")
#             feature = feature_extractor[2](feature)
#             # print(f"After bottleneck: {feature.shape}")

#             features.append(feature.cpu())



#     return torch.cat(features, dim=0)


def collect_feature(dataloader, feature_extractor, device):
    features = []
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(dataloader, desc="Extracting Features", leave=True)):
            # print(type(data))
            inputs = data[0].to(device)  # (B, 3, 224, 224)
            # inputs = torch.stack(data[0]).to(device)  # (B, 3, 224, 224)
            # inputs = inputs.view(-1, *inputs.shape[2:])  # Reshape to [2*36, 3, 224, 224]
            # Extract CNN features
            feature = feature_extractor[0](inputs)  # Convolutional backbone
            # print(f"After CNN: {feature.shape}")  # Should be (B, C, H, W)

            if(feature.dim()==2):
                feature = feature.unsqueeze(-1).unsqueeze(-1)  # Converts (B, C) → (B, C, 1, 1)

            # Apply pooling
            feature = feature_extractor[1](feature)
            # print(f"After pooling: {feature.shape}")  # Should be (B, C, 1, 1)

            # Flatten before bottleneck
            feature = feature.view(feature.shape[0], -1)  # (B, C)
            # print(f"After flattening: {feature.shape}")  # Should be (B, 2048)

            # Apply bottleneck
            if(len(feature_extractor)==3):
                feature = feature_extractor[2](feature)
            # print(f"After bottleneck: {feature.shape}")  # Should be (B, bottleneck_dim)

            features.append(feature.cpu())

    return torch.cat(features, dim=0)  # Shape (N, bottleneck_dim)
