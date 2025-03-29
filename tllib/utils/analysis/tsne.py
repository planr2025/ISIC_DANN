import torch
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for remote servers

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

# def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
#               filename: str, source_color='red', target_color='blue',
#               source_label='Source Domain', target_label='Target Domain'):
#     source_feature = source_feature.numpy()
#     target_feature = target_feature.numpy()
#     features = np.concatenate([source_feature, target_feature], axis=0)

#     X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

#     domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)

#     scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, 
#                            cmap=col.ListedColormap([target_color, source_color]), s=20)

#     plt.xlabel("t-SNE Dimension 1", fontsize=12)
#     plt.ylabel("t-SNE Dimension 2", fontsize=12)
#     plt.title("t-SNE Visualization of Source & Target Domains", fontsize=14)

#     # Ensure the legend is properly displayed
#     legend_elements = [
#         plt.Line2D([0], [0], marker='o', color='w', markersize=10, 
#                    markerfacecolor=source_color, label=f"{source_label} (source)"),
#         plt.Line2D([0], [0], marker='o', color='w', markersize=10, 
#                    markerfacecolor=target_color, label=f"{target_label} (target)")
#     ]
#     legend = plt.legend(handles=legend_elements, loc='best', fontsize=12, frameon=True)

#     plt.draw()  # Forces Matplotlib to update the figure
#     plt.tight_layout()  # Adjusts layout to prevent overlapping elements
#     plt.savefig(filename)  # Save the plot

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from sklearn.manifold import TSNE
import torch

# def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
#              source_labels: np.ndarray, target_labels: np.ndarray,
#               filename: str, source_label, target_label):
    
#     # Convert tensors to numpy
#     source_feature = source_feature.numpy()
#     target_feature = target_feature.numpy()

#     # Combine features
#     features = np.concatenate([source_feature, target_feature], axis=0)
    
#     # Apply t-SNE
#     X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
    
#     # Separate transformed features
#     source_tsne = X_tsne[:len(source_feature)]
#     target_tsne = X_tsne[len(source_feature):]

#     # Class mapping for better visualization
#     class_mapping = {0: "Benign", 1: "Malignant"}
#     colors = {0: ('lightcoral', 'lightskyblue'),  # Benign: Light Red (Source), Light Blue (Target)
#               1: ('red', 'blue')}  # Malignant: Dark Red (Source), Dark Blue (Target)

#     fig, ax = plt.subplots(figsize=(10, 10))
#     # print(source_labels)
#     # Plot source domain points
#     for class_idx in np.unique(source_labels):
#         idx = np.where(source_labels == class_idx)[0]
#         ax.scatter(source_tsne[idx, 0], source_tsne[idx, 1], 
#                    color=colors[class_idx][0], label=f"{source_label} - {class_mapping[class_idx]}", 
#                    alpha=0.6, s=20)
    
#     # Plot target domain points
#     for class_idx in np.unique(target_labels):
#         idx = np.where(target_labels == class_idx)[0]
#         ax.scatter(target_tsne[idx, 0], target_tsne[idx, 1], 
#                    color=colors[class_idx][1], label=f"{target_label} - {class_mapping[class_idx]}", 
#                    alpha=0.6, s=20)

#     # Remove unnecessary spines
#     for spine in ax.spines.values():
#         spine.set_visible(False)

#     # Labels & Title
#     plt.xlabel("t-SNE Dimension 1", fontsize=12)
#     plt.ylabel("t-SNE Dimension 2", fontsize=12)
#     plt.title("t-SNE Visualization of Source & Target Domains", fontsize=14)

#     # Create legend with class names and domains
#     legend_elements = [
#         mlines.Line2D([0], [0], marker='o', color='w', markersize=10, 
#                       markerfacecolor=colors[0][0], label=f"{source_label} - Benign"),
#         mlines.Line2D([0], [0], marker='o', color='w', markersize=10, 
#                       markerfacecolor=colors[1][0], label=f"{source_label} - Malignant"),
#         mlines.Line2D([0], [0], marker='o', color='w', markersize=10, 
#                       markerfacecolor=colors[0][1], label=f"{target_label} - Benign"),
#         mlines.Line2D([0], [0], marker='o', color='w', markersize=10, 
#                       markerfacecolor=colors[1][1], label=f"{target_label} - Malignant")
#     ]
#     plt.legend(handles=legend_elements, loc='best', fontsize=12, frameon=True)

#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.show()

def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
             source_labels: np.ndarray, target_labels: np.ndarray,
              filename: str, source_label, target_label):

    # Convert tensors to numpy
    source_feature = source_feature.cpu().numpy()
    target_feature = target_feature.cpu().numpy()

    # Combine features
    features = np.concatenate([source_feature, target_feature], axis=0)

    # Apply t-SNE with better parameters
    X_tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, learning_rate=200, random_state=33).fit_transform(features)

    # Separate transformed features
    source_tsne = X_tsne[:len(source_feature)]
    target_tsne = X_tsne[len(source_feature):]

    print(f"Source t-SNE Shape: {source_tsne.shape}")
    print(f"Target t-SNE Shape: {target_tsne.shape}")

    # Class mapping for better visualization
    class_mapping = {0: "Negative", 1: "Positive"}
    markers = {0: 'o', 1: 's'}  # Circles for Negative, Diamonds for Positive
    colors = {'source': 'red', 'target': 'blue'}  # Same color for all classes

    fig, ax = plt.subplots(figsize=(14, 14))

    # Plot target domain points
    for class_idx in np.unique(target_labels):
        print(class_idx)
        idx_list = np.where(target_labels == class_idx)[0]
        for i in idx_list:  # Iterate over each index
            ax.scatter(target_tsne[i, 0], target_tsne[i, 1], 
                    color=colors['target'], edgecolor='black', marker=markers[class_idx],
                    label=f"{target_label} - {class_mapping[class_idx]}", 
                    alpha=0.8, s=50)

    # Plot source domain points
    for class_idx in np.unique(source_labels):
        idx_list = np.where(source_labels == class_idx)[0]
        for i in idx_list:  # Iterate over each index
            ax.scatter(source_tsne[i, 0], source_tsne[i, 1], 
                    color=colors['source'], edgecolor='black', marker=markers[class_idx],
                    label=f"{source_label} - {class_mapping[class_idx]}", 
                    alpha=0.8, s=50)

    # Remove unnecessary spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Labels & Title
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.title(f"t-SNE Visualization of Source-{source_label} & Target-{target_label} Domains", fontsize=16, fontweight='bold')

    # Improved Legend
    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', markersize=12, 
                      markerfacecolor=colors['source'], markeredgecolor='black', label=f"{source_label} - Negative"),
        mlines.Line2D([0], [0], marker='s', color='w', markersize=12, 
                      markerfacecolor=colors['source'], markeredgecolor='black', label=f"{source_label} - Positive"),
        mlines.Line2D([0], [0], marker='o', color='w', markersize=12, 
                      markerfacecolor=colors['target'], markeredgecolor='black', label=f"{target_label} - Negative"),
        mlines.Line2D([0], [0], marker='s', color='w', markersize=12, 
                      markerfacecolor=colors['target'], markeredgecolor='black', label=f"{target_label} - Positive")
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=14, frameon=True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # Save with high resolution
    plt.show()
