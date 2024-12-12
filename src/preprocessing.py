import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_data_loaders(batch_size, num_workers, num_clients):
    # Transformations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load CIFAR-100 dataset
    dataset_train = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
    dataset_test = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)

    # Split the training dataset into smaller subsets for each client
    dataset_size = len(dataset_train)
    subset_size = dataset_size // num_clients
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    client_datasets = []
    for i in range(num_clients):
        subset_indices = indices[i * subset_size: (i + 1) * subset_size]
        client_dataset = Subset(dataset_train, subset_indices)
        client_datasets.append(client_dataset)

    # Create DataLoaders for each client
    client_loaders = []
    for client_dataset in client_datasets:
        client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        client_loaders.append(client_loader)

    # Test DataLoader (common for all clients)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return client_loaders, test_loader

if __name__ == "__main__":
    batch_size = 256
    num_workers = 4
    num_clients = 10

    client_loaders, test_loader = get_data_loaders(batch_size, num_workers, num_clients)

    # Checking the first client loader to see if everything works correctly
    for images, labels in client_loaders[0]:
        print(f'Batch size: {images.size(0)}')
        break

    print("Preprocessing for federated learning completed successfully.")