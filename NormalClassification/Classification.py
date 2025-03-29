import sys
import os
# Get the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir) 

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import utils
from tllib.utils.data import ForeverDataIterator
from datetime import datetime
from tqdm import tqdm

# Define argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model on a dataset.")
    parser.add_argument('--arch', type=str, required=True, help='Model architecture (e.g., resnet101)')
    parser.add_argument('--data', type=str, required=True, help='Name of the dataset (e.g., DomainNet)')
    parser.add_argument('--root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--source', type=str, nargs='+', required=True, help='Source domain(s)')
    parser.add_argument('--target', type=str, nargs='+', required=True, help='Target domain(s)')
    parser.add_argument('--scratch', action='store_true', help='Train from scratch (no pretraining)')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save logs and outputs')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--iters_per_epoch', type=int, default=500, help='Number of iterations per epoch')
    parser.add_argument('--phase', type=str, default="train", help='Phase of training')
    return parser.parse_args()

# Modify the classification layer
class FineTuneModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super(FineTuneModel, self).__init__()
        self.backbone = backbone
        self.fc = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.backbone.fc = nn.Identity()  # Remove the original fully connected layer

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)
    


# Training and evaluation function
# Detect GPU (or fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import torch
from tqdm import tqdm
from datetime import datetime
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import os
import torch
from datetime import datetime
from tqdm import tqdm
import utils

def train_and_evaluate(model, train_iter, val_loader, criterion, optimizer, num_epochs, output_dir, args,mode= 'mode1', phase='train'):
    log_file = os.path.join(output_dir, f'{mode}_{phase}_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    if phase == 'train':
        with open(log_file, 'w') as f:
            f.write(f"Training Log\n")
            f.write(f"Date: {datetime.now()}\n\n")
        
        model.to(device)
        best_acc = 0.0

        for epoch in range(num_epochs):
            model.train()
            progress_bar = tqdm(range(args.iters_per_epoch), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

            total_loss = 0
            correct = 0
            total = 0

            for _ in progress_bar:
                inputs, labels = next(train_iter)[:2]
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = total_loss / len(train_iter.data_loader)
            epoch_acc = correct / total * 100

            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%\n")

            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
         # ✅ Move model to eval mode before validation
        model.eval()
        model.to(device)

        # ✅ Validate model
        acc1, precision, recall, f1 = utils.validate(val_loader, model, args, device)

        # ✅ Save best model based on validation accuracy
        if acc1 > best_acc:
            best_acc = acc1
            best_model_path = os.path.join(output_dir,   f'best_model_{mode}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 Best model saved at {best_model_path} with Accuracy: {acc1:.2f}%")
        final_model_path = os.path.join(output_dir, f'final_model_{mode}.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"🎯 Final trained model saved at {final_model_path}")
    
    elif phase == 'validate':
        validation_output_dir = os.path.join(output_dir, "validation_results")
        os.makedirs(validation_output_dir, exist_ok=True)

        val_log_file = os.path.join(validation_output_dir, f'validation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        best_model_path = os.path.join(output_dir, f'best_model_{mode}.pth')

        
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print(f"✅ Loaded best model from {best_model_path}")
        else:
            print("❌ Best model not found! Validation aborted.")
            return
        
        model.eval()
        model.to(device)
        
        acc1, precision, recall, f1 = utils.validate(val_loader, model, args, device)

        with open(val_log_file, 'w') as f:
            f.write(f"Validation Metrics:\n")
            f.write(f"Accuracy: {acc1:.2f}%\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")

        print(f"✅ Validation - Accuracy: {acc1:.2f}% | F1 Score: {f1:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f}")

        # Save Confusion Matrix
        compute_confusion_matrix(model, val_loader, validation_output_dir, device, f"{mode}_validation")
        print(f"📊 Confusion Matrix saved for validation phase")

def compute_confusion_matrix(model, val_loader, output_dir, device, mode, class_names=None):
    """
    Computes the confusion matrix for the given model and validation dataloader.
    Saves the confusion matrix as a JPG file.
    """
    model.eval()  # Set model to evaluation mode
    model.to(device)

    y_true = []
    y_pred = []

    with torch.no_grad():  # No need to compute gradients during validation
        for i, data in enumerate(val_loader):
            images, target = data[:2]
            images = images.to(device)
            target = target.to(device)

            # Compute output
            output = model(images)
            if isinstance(output, tuple):  # If model outputs a tuple
                logits, _ = output  # Extract logits
            else:
                logits = output
                logits = logits.squeeze()

            # Store predictions and ground truth
            y_true.extend(target.cpu().numpy())  # Convert to NumPy
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())  # Get predicted class

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Define class labels
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]  # Use indices if class names are not provided

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(cm, index=class_names, columns=class_names), 
                annot=True, fmt='d', cmap="Blues", cbar=True)
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # Save the confusion matrix image
    cm_path = os.path.join(output_dir, f'confusion_matrix_{mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Confusion matrix saved at {cm_path}")

    return cm
# Main function
def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the backbone model
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)

    # Modify the model for fine-tuning
    model = FineTuneModel(backbone, args.num_classes)

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio,
                                                random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)

    # Load datasets using get_dataset
    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)

    # Define dataloaders
    train_loader1 = DataLoader(train_source_dataset, batch_size=args.batch_size, num_workers=args.workers,  shuffle=True)
    val_loader2 = DataLoader(train_target_dataset, batch_size=args.batch_size, num_workers=args.workers,  shuffle=False)

    print(train_loader1)
    # Create ForeverDataIterator for training
    train_iter1 = ForeverDataIterator(train_loader1)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Mode 1: Train on dataset1 and validate on dataset2
    train_and_evaluate(model, train_iter1, val_loader2, criterion, optimizer, args.epochs, args.output_dir,args, mode='mode1', phase=args.phase)

    # # Mode 2: Train and test on dataset1 only
    # train_and_evaluate(model, train_iter1, train_loader1, criterion, optimizer, args.epochs, args.output_dir,args, mode='mode2', phase=args.phase)  

if __name__ == '__main__':
    main()