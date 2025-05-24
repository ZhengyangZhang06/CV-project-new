import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset for FER2013
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

# Function to load and preprocess the FER2013 dataset
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
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ResNet Basic Block
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

# ResNet Model
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

# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

# Main training function
def train_fer2013_resnet():
    # Make sure the directory exists for model checkpoints
    os.makedirs('models', exist_ok=True)
    start_time = time.time()
    
    # Load the dataset
    csv_path = 'fer2013/train.csv'  # Adjust path if needed
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fer2013(csv_path)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Count samples per class before augmentation
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    print("\nClass distribution before augmentation:")
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    for i, count in zip(unique_classes, class_counts):
        print(f"{emotion_labels[i]}: {count}")
    
    # Targeted augmentation for "Disgust" class (label 1)
    disgust_indices = np.where(y_train == 1)[0]
    disgust_samples = X_train[disgust_indices]
    disgust_labels = y_train[disgust_indices]
    print(f"\nFound {len(disgust_indices)} 'Disgust' samples for augmentation")
    
    # Define a strong augmentation pipeline for disgust samples
    disgust_augmentations = [
        # Horizontal flip
        lambda img: np.fliplr(img.reshape(IMG_SIZE, IMG_SIZE)).flatten(),
        # Rotation +15 degrees
        lambda img: np.rot90(img.reshape(IMG_SIZE, IMG_SIZE), k=1).flatten(),
        # Rotation -15 degrees
        lambda img: np.rot90(img.reshape(IMG_SIZE, IMG_SIZE), k=3).flatten(),
        # Slight zoom in
        lambda img: img,  # Placeholder, will be handled by dataset transform
        # Brightness variation
        lambda img: np.clip(img * 1.2, 0, 255),
        # Contrast variation
        lambda img: np.clip((img - 128) * 1.3 + 128, 0, 255),
    ]
    
    # Generate augmented samples
    augmented_disgust_samples = []
    augmented_disgust_labels = []
    
    for i, sample in enumerate(disgust_samples):
        for aug_func in disgust_augmentations:
            augmented_sample = aug_func(sample)
            augmented_disgust_samples.append(augmented_sample)
            augmented_disgust_labels.append(1)  # Label 1 for Disgust
    
    # Visualize some example augmentations
    def visualize_augmentations(samples, aug_functions):
        # Select up to 3 random samples for visualization
        num_samples = min(3, len(samples))
        if num_samples > 0:
            sample_indices = np.random.choice(len(samples), num_samples, replace=False)
            
            # Create figure with subplots
            fig, axes = plt.subplots(num_samples, len(aug_functions) + 1, figsize=(15, 3 * num_samples))
            if num_samples == 1:
                axes = axes.reshape(1, -1)
                
            aug_names = ['Original', 'Horizontal Flip', 'Rotation +90°', 
                         'Rotation -90°', 'No Change', 'Brightness +20%', 'Contrast +30%']
            
            # Plot images
            for i, idx in enumerate(sample_indices):
                sample_img = samples[idx].reshape(IMG_SIZE, IMG_SIZE)
                
                # Original image
                axes[i, 0].imshow(sample_img, cmap='gray')
                axes[i, 0].set_title(aug_names[0])
                axes[i, 0].axis('off')
                
                # Augmented images
                for j, aug_func in enumerate(aug_functions):
                    aug_img = aug_func(samples[idx]).reshape(IMG_SIZE, IMG_SIZE)
                    axes[i, j+1].imshow(aug_img, cmap='gray')
                    axes[i, j+1].set_title(aug_names[j+1])
                    axes[i, j+1].axis('off')
            
            plt.tight_layout()
            plt.savefig('disgust_augmentation_examples.png')
            plt.show()
            print("Augmentation examples saved to 'disgust_augmentation_examples.png'")
    
    # Visualize augmentations on a few disgust samples
    print("\nVisualizing augmentation examples:")
    visualize_augmentations(disgust_samples, disgust_augmentations)
    
    # Add augmented samples to the training set
    X_train = np.vstack([X_train, np.array(augmented_disgust_samples)])
    y_train = np.append(y_train, np.array(augmented_disgust_labels))
    
    # Count samples per class after augmentation
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    print("\nClass distribution after augmentation:")
    for i, count in zip(unique_classes, class_counts):
        print(f"{emotion_labels[i]}: {count}")
    
    # Data augmentation for training (still applies to all samples during training)
    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, translate=(0.2, 0.2)),
        transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.8, 1.0)),
    ])
    
    # Create datasets
    train_dataset = FER2013Dataset(X_train, y_train, transform=train_transform)
    val_dataset = FER2013Dataset(X_val, y_val)
    test_dataset = FER2013Dataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Create the model, loss function, and optimizer
    model = ResNet18().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6, verbose=True)
    
    # Print model summary
    print(model)
    
    # Training loop
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/fer2013_resnet_best.pth')
            print(f"Epoch {epoch+1}/{EPOCHS} - Saved best model with val_acc: {val_acc:.2f}%")
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f}%, "
              f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.2f}%")
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('models/fer2013_resnet_best.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    print(f"Test accuracy: {test_acc:.2f}%")
    
    # Save the final model
    torch.save(model.state_dict(), 'models/fer2013_resnet_final.pth')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('fer2013_resnet_training.png')
    plt.show()
    
    # report total elapsed time
    elapsed = time.time() - start_time
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Total training time: {int(hrs)}h {int(mins)}m {secs:.2f}s")
    
    return model

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    train_fer2013_resnet()