import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend first

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse

# Constants
IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion labels
EMOTIONS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# Custom Dataset for FER2013 (same as in train.py)
class FER2013Dataset(Dataset):
    def __init__(self, pixels, labels, transform=None):
        self.pixels = pixels
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.pixels[idx].reshape(IMG_SIZE, IMG_SIZE).astype(np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# Function to load and preprocess the FER2013 dataset (same as in train.py)
def load_fer2013(csv_file):
    data = pd.read_csv(csv_file)
    
    # Extract pixels and labels
    pixels = data['pixels'].apply(lambda x: np.array(x.split(' ')).astype('float32'))
    X = np.stack(pixels.values)
    
    # Extract emotion labels
    y = data['emotion'].values
    
    # Split data based on 'Usage' column if it exists
    if 'Usage' in data.columns:
        train_data = data[data['Usage'] == 'Training']
        val_data = data[data['Usage'] == 'PublicTest']
        test_data = data[data['Usage'] == 'PrivateTest']
        
        X_train = np.stack(train_data['pixels'].apply(lambda x: np.array(x.split(' ')).astype('float32')))
        y_train = train_data['emotion'].values
        
        X_val = np.stack(val_data['pixels'].apply(lambda x: np.array(x.split(' ')).astype('float32')))
        y_val = val_data['emotion'].values
        
        X_test = np.stack(test_data['pixels'].apply(lambda x: np.array(x.split(' ')).astype('float32')))
        y_test = test_data['emotion'].values
    else:
        # If 'Usage' column doesn't exist, manually split the data
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ResNet Basic Block (same as in train.py)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ResNet Model (same as in train.py)
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=NUM_CLASSES):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Create ResNet18 model
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# Evaluation function
def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    return accuracy, all_preds, all_targets

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[EMOTIONS[i] for i in range(NUM_CLASSES)],
                yticklabels=[EMOTIONS[i] for i in range(NUM_CLASSES)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

# Function to plot all classification reports in one figure
def plot_combined_classification_reports(train_true, train_pred, val_true, val_pred, test_true, test_pred):
    plt.figure(figsize=(24, 8))
    
    # Generate reports
    datasets = [
        ("Training", train_true, train_pred, 1),
        ("Validation", val_true, val_pred, 2),
        ("Test", test_true, test_pred, 3)
    ]
    
    for name, y_true, y_pred, pos in datasets:
        # Generate classification report
        report = classification_report(
            y_true, y_pred,
            target_names=[EMOTIONS[i] for i in range(NUM_CLASSES)],
            digits=3,
            output_dict=True
        )
        
        # Process report data
        report_df = pd.DataFrame(report).T
        report_df = report_df.drop('support', axis=1)
        report_df = report_df.drop('accuracy', axis=0)
        
        # Create subplot
        plt.subplot(1, 3, pos)
        sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.3f', vmin=0.0, vmax=1.0)
        plt.title(f'{name} Metrics')
    
    plt.tight_layout()
    plt.savefig('combined_metrics.png')
    print("Saved combined metrics visualization to combined_metrics.png")

# Function to plot all confusion matrices in one figure
def plot_combined_confusion_matrices(train_true, train_pred, val_true, val_pred, test_true, test_pred):
    plt.figure(figsize=(24, 8))
    
    # Generate confusion matrices
    datasets = [
        ("Training", train_true, train_pred, 1),
        ("Validation", val_true, val_pred, 2),
        ("Test", test_true, test_pred, 3)
    ]
    
    for name, y_true, y_pred, pos in datasets:
        cm = confusion_matrix(y_true, y_pred)
        
        plt.subplot(1, 3, pos)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[EMOTIONS[i] for i in range(NUM_CLASSES)],
                    yticklabels=[EMOTIONS[i] for i in range(NUM_CLASSES)])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{name} Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('combined_confusion_matrices.png')
    print("Saved combined confusion matrices to combined_confusion_matrices.png")

# Function to display sample predictions
def display_sample_predictions(model, test_dataset, device, num_samples=5):
    # Get random indices
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    model.eval()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, label = test_dataset[idx]
            img = img.unsqueeze(0).to(device)  # Add batch dimension
            output = model(img)
            _, predicted = output.max(1)
            
            # Convert image for display
            img_display = img.cpu().squeeze().numpy()
            
            axes[i].imshow(img_display, cmap='gray')
            axes[i].set_title(f"True: {EMOTIONS[label.item()]}\nPred: {EMOTIONS[predicted.item()]}")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()

def plot_dataset_distribution(y_train, y_val, y_test):
    """
    Plot the distribution of labels across training, validation, and test sets.
    
    Parameters:
    y_train, y_val, y_test: Arrays of labels for each dataset
    """
    plt.figure(figsize=(20, 15))
    
    # Get counts for each dataset
    datasets = [
        ("Training", y_train, 1),
        ("Validation", y_val, 2),
        ("Test", y_test, 3),
        ("Combined", np.concatenate([y_train, y_val, y_test]), 4)
    ]
    
    # Set up a color palette
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    
    for name, labels, pos in datasets:
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        
        # Create subplot
        plt.subplot(2, 2, pos)
        bars = plt.bar(range(NUM_CLASSES), counts, color=colors)
        
        # Add count labels on top of each bar
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{count}', ha='center', va='bottom')
        
        plt.title(f'{name} Set: {sum(counts)} samples')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(range(NUM_CLASSES), [EMOTIONS[i] for i in range(NUM_CLASSES)], rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add percentage labels inside or next to each bar
        for i, (bar, count) in enumerate(zip(bars, counts)):
            percentage = count / sum(counts) * 100
            plt.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{percentage:.1f}%', ha='center', va='center', 
                    color='black' if percentage < 15 else 'white',
                    fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png')
    print("Saved dataset distribution visualization to dataset_distribution.png")
    
    # Return counts for reporting
    return {
        "train": np.bincount(y_train, minlength=NUM_CLASSES),
        "val": np.bincount(y_val, minlength=NUM_CLASSES),
        "test": np.bincount(y_test, minlength=NUM_CLASSES),
        "total": np.bincount(np.concatenate([y_train, y_val, y_test]), minlength=NUM_CLASSES)
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate FER2013 model')
    parser.add_argument('--model_path', type=str, default='models/fer2013_resnet_best.pth',
                        help='Path to the trained model')
    parser.add_argument('--data_path', type=str, default='fer2013/train.csv',
                        help='Path to the dataset CSV file')
    args = parser.parse_args()

    # Check if model exists
    if not os.path.isfile(args.model_path):
        print(f"Error: Model file {args.model_path} not found")
        return

    print(f"Using device: {DEVICE}")
    
    # Load all datasets
    print("Loading datasets...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fer2013(args.data_path)
    
    # Add this after loading datasets but before creating dataset objects
    print("Analyzing dataset distribution...")
    label_counts = plot_dataset_distribution(y_train, y_val, y_test)
    
    # Print a detailed summary
    print("\n===== Dataset Label Distribution =====")
    for i in range(NUM_CLASSES):
        emotion = EMOTIONS[i]
        train_count = label_counts["train"][i]
        val_count = label_counts["val"][i]
        test_count = label_counts["test"][i]
        total_count = label_counts["total"][i]
        
        print(f"{emotion}: Total={total_count} (Train={train_count}, Val={val_count}, Test={test_count})")
    
    # Create datasets and dataloaders for each split
    train_dataset = FER2013Dataset(X_train, y_train)
    val_dataset = FER2013Dataset(X_val, y_val)
    test_dataset = FER2013Dataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Create and load the model
    print("Loading model...")
    model = ResNet18().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path))
    
    # Evaluate on all datasets
    print("\n===== Evaluating on Training Data =====")
    train_accuracy, train_preds, train_targets = evaluate(model, train_loader, DEVICE)
    print(f"Training accuracy: {train_accuracy:.2f}%")
    
    print("\n===== Evaluating on Validation Data =====")
    val_accuracy, val_preds, val_targets = evaluate(model, val_loader, DEVICE)
    print(f"Validation accuracy: {val_accuracy:.2f}%")
    
    print("\n===== Evaluating on Test Data =====")
    test_accuracy, test_preds, test_targets = evaluate(model, test_loader, DEVICE)
    print(f"Test accuracy: {test_accuracy:.2f}%")
    
    # Generate individual reports and matrices (unchanged)
    print("\nTraining Classification Report:")
    train_report = classification_report(train_targets, train_preds, 
                                       target_names=[EMOTIONS[i] for i in range(NUM_CLASSES)],
                                       digits=3)
    print(train_report)
    
    print("\nValidation Classification Report:")
    val_report = classification_report(val_targets, val_preds, 
                                     target_names=[EMOTIONS[i] for i in range(NUM_CLASSES)],
                                     digits=3)
    print(val_report)
    
    print("\nTest Classification Report:")
    test_report = classification_report(test_targets, test_preds, 
                                      target_names=[EMOTIONS[i] for i in range(NUM_CLASSES)],
                                      digits=3)
    print(test_report)
    
    # Generate combined visualizations
    print("Generating combined visualizations...")
    plot_combined_classification_reports(train_targets, train_preds, 
                                        val_targets, val_preds, 
                                        test_targets, test_preds)
    
    plot_combined_confusion_matrices(train_targets, train_preds,
                                    val_targets, val_preds,
                                    test_targets, test_preds)
    

if __name__ == "__main__":
    main()
