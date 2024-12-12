import os
import torch
import argparse
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, ConcatDataset, Dataset
from collections import Counter, defaultdict
from tqdm import tqdm
import torch.nn as nn
from PIL import Image
import random


class CustomPlasticDataset(Dataset):
    def __init__(self, root_dir, class_mapping, transform, tetra_transform=None, ps_transform=None, pvc_transform=None, diverse_transform=None):
        self.root_dir = root_dir
        self.class_mapping = class_mapping
        self.transform = transform
        self.tetra_transform = tetra_transform
        self.ps_transform = ps_transform
        self.pvc_transform = pvc_transform
        self.diverse_transform = diverse_transform
        self.image_paths = []
        self.labels = []
        
        # Gather image paths and labels
        for class_folder in class_mapping.keys():
            # Load original images
            image_dir = os.path.join(root_dir, class_folder, 'images_cutout')
            if os.path.exists(image_dir):
                image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
                print(f"Loaded {len(image_files)} original images for {class_folder}")
                self.image_paths.extend([os.path.join(image_dir, img) for img in image_files])
                self.labels.extend([class_mapping[class_folder]] * len(image_files))

            # Load augmented images if they exist
            augmented_dir = os.path.join(root_dir, class_folder)  # Path to the augmented class
            if os.path.exists(augmented_dir) and 'Augmented' in class_folder:
                augmented_files = [f for f in os.listdir(augmented_dir) if f.endswith(('.jpg', '.png'))]
                print(f"Loaded {len(augmented_files)} augmented images for {class_folder}")
                self.image_paths.extend([os.path.join(augmented_dir, img) for img in augmented_files])
                self.labels.extend([class_mapping[class_folder]] * len(augmented_files))  # Map to the same class label
                
        print(f"Unique labels in the dataset: {set(self.labels)}")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB to handle all image modes
        label = self.labels[idx]
        
        # Apply specific transformations based on class
        if label == 3:  # Tetra
            image = self.tetra_transform(image) if self.tetra_transform else self.transform(image)
        elif label == 4:  # PS
            image = self.ps_transform(image) if self.ps_transform else self.transform(image)
        elif label == 5:  # PVC
            image = self.pvc_transform(image) if self.pvc_transform else self.transform(image)
        else:
            # Use diverse_transform with a 50% chance for other classes
            if self.diverse_transform is not None and random.random() > 0.5:
                image = self.diverse_transform(image)
            else:
                image = self.transform(image)
        
        return image, label
# Augmentation for TETRA (Moderate)
tetra_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Augmentation for PS (Light)
ps_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Regular transform for other classes
regular_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
])
    
# Augmentation for PVC (Light)
pvc_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define more diverse transformations for augmentation
diverse_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
    transforms.RandomRotation(degrees=30),  # Random rotation by up to 30 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color adjustments
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Random crop with resizing
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    

class_mapping = {
    #BigBag2
    'BigBag2_1_PET': 0,  # PET
    'BigBag2_2_PP': 1,   # PP
    'BigBag2_3_PE': 2,   # PE
    'BigBag2_4_Tetra': 3, # Tetra
    'BigBag2_5_PVC': 5, # PVC
    'BigBag2_6_PS': 4,   # PS
    'BigBag2_7_Other': 6, # Other
    'BigBag2_4_Tetra_Augmented': 3,  # Augmented Tetra
    'BigBag2_6_PS_Augmented': 4,  # Augmented PS
    
    #BigBag4
    'BigBag4_1_PET': 0,  # PET
    'BigBag4_2_PP': 1,   # PP
    'BigBag4_3_PE': 2,   # PE
    'BigBag4_4_Tetra': 3, # Tetra
    'BigBag4_6_PS': 4,   # PS
    'BigBag4_5_PVC': 5, # PVC
    'BigBag4_7_Other': 6, # Other
    
    #BigBag1
    'BigBag1_1_PET': 0,  # PET
    'BigBag1_2_PP': 1,   # PP
    'BigBag1_3_PE': 2,   # PE
    'BigBag1_4_Tetra': 3, # Tetra
    #'BigBag2_4_Tetra_Augmented': 3,  # Augmented Tetra
    #'BigBag2_5_PVC': 5, # PVC
    'BigBag1_6_PS': 4,   # PS
    'BigBag1_7_Other': 6, # Other
    #'BigBag2_6_PS_Augmented': 4,  # Augmented PS
    
    #BigBag3
    'BigBag3_PET': 0,  # PET
    'BigBag3_2_PP': 1,   # PP
    'BigBag3_PE': 2,   # PE
    'BigBag3_TETRA': 3, # Tetra
    #'BigBag3_PVC': 5, # PVC
    'BigBag3_6_PS': 4,   # PS
    'BigBag3_Other': 6, # Other
    
    'DWRL7_extension_2_PVC': 5, # 
    'BigBag2_5_PVC_Augmented': 5,  # Augmented PVC
}



# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Argument parser for configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Unified script for CIFAR-100 and DWRL Federated Learning")
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar", "dwrl"], help="Dataset to use: cifar or dwrl")
    parser.add_argument("--num_clients", type=int, default=8, help="Number of clients")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--use_replay", action="store_true", help="Enable replay strategy for DWRL dataset")
    return parser.parse_args()

# Dataset preparation
def prepare_dataset(dataset_name, transform, num_clients, use_replay=False):
    if dataset_name == "cifar":
        dataset_train = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    elif dataset_name == "dwrl":
        dataset_train = CustomPlasticDataset(
            root_dir="Paper3_FCL/data/data_DWRL7", 
            class_mapping=class_mapping, 
            transform=transform,
            tetra_transform=tetra_transform,
            ps_transform=ps_transform,
            pvc_transform=pvc_transform,
            diverse_transform=diverse_transform
        )
        print(f"Dataset loaded: {len(dataset_train)} samples")
        dataset_test = None  # Adjust based on setup

    train_dataset, val_dataset = None, None
    if dataset_name == "cifar":
        num_train_images = int(0.9 * len(dataset_train))
        num_val_images = len(dataset_train) - num_train_images
        train_dataset, val_dataset = random_split(dataset_train, [num_train_images, num_val_images])

    client_splits = random_split(train_dataset, [len(train_dataset) // num_clients] * num_clients)
    return client_splits, val_dataset, dataset_test

# Model preparation
def prepare_model(num_classes=7, use_dropout=False, dropout_prob=0.2):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_prob),
        nn.Linear(num_ftrs, num_classes)
    ) if use_dropout else nn.Linear(num_ftrs, num_classes)
    return model

# Train model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Train Accuracy: {train_accuracy:.2f}%')

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.2f}%')

    return model

# Federated learning
def federated_learning(client_models, train_loaders, val_loaders, num_clients, num_epochs):
    global_model = prepare_model().to(device)
    for round in range(num_epochs):
        print(f'\n--- Federated Learning Round {round + 1} ---')
        client_state_dicts = []

        for client_idx in range(num_clients):
            print(f'\nTraining client {client_idx + 1}')
            client_model = client_models[client_idx]
            train_loader = train_loaders[client_idx]
            val_loader = val_loaders[client_idx]
            optimizer = torch.optim.Adam(client_model.parameters(), lr=0.0001)
            criterion = nn.CrossEntropyLoss()

            client_model = train_model(client_model, train_loader, val_loader, criterion, optimizer, num_epochs=1)
            client_state_dicts.append(client_model.state_dict())

        global_state_dict = {key: sum(d[key] for d in client_state_dicts) / len(client_state_dicts) for key in client_state_dicts[0]}
        global_model.load_state_dict(global_state_dict)

    return global_model

# Test model
def test_model(model, test_loader):
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.2f}%')

# Main function
def main():
    args = parse_args()

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    client_splits, val_dataset, test_dataset = prepare_dataset(args.dataset, transform, args.num_clients, args.use_replay)

    client_models = [prepare_model(args.num_classes).to(device) for _ in range(args.num_clients)]
    train_loaders = [DataLoader(client, batch_size=args.batch_size, shuffle=True) for client in client_splits]
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) if test_dataset else None

    global_model = federated_learning(client_models, train_loaders, [val_loader] * args.num_clients, args.num_clients, args.num_epochs)

    if test_loader:
        test_model(global_model, test_loader)

if __name__ == "__main__":
    main()
