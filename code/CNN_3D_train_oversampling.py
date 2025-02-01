import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from CNN_3D import Simple3DRegressionCNN
from dataset_loader_3D import CustomMRIDataset
import logging


# Instantiate the dataset
dataset = CustomMRIDataset(csv_file='../tor/paths_labels.csv', img_dir='../tor/rel3_dhcp_anat_pipeline/', filter_size=(2, 2, 2))
img, label = dataset[0]
print(img.shape)
print(label)
torch.save(dataset, 'dataset.pt')

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the CNN model
model = Simple3DRegressionCNN()
model = model.to(device)

# Train/test/validation split
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Get the labels from the train_datasset (for weighted sampling)
train_labels = [label.item() for _, label in train_dataset]

# Calculate the weights
counts = np.bincount(train_labels)
weights = 1.0 / (counts + 1e-10)
sample_weights = weights[np.array(train_labels).astype(int)]

# Create a sampler
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Create data loaders for batching the data and assign sampler to train_loader
train_loader = DataLoader(train_dataset, batch_size=2, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)

# Create an empty list to store the labels
labels_list = []

# Iterate over the DataLoader
for images, labels in train_loader:
    labels_list.extend(labels.cpu().numpy().tolist())

# Define the bins
bins = np.arange(0, 101, 10)

# Use np.histogram to categorize labels into bins
counts, bin_edges = np.histogram(labels_list, bins=bins)

# Format the intervals as strings
categories = [f'({int(bin_edges[i])}, {int(bin_edges[i+1])}]' for i in range(len(bin_edges)-1)]

# Plot the counts
plt.figure(figsize=(10, 6))
plt.bar(categories, counts)
plt.title('Label Distribution After Oversampling')
plt.xlabel('Label Category')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Save the plot as a PNG file
plt.savefig('label_distribution.png', bbox_inches='tight')
plt.close()

# Create a label tensor
label = torch.tensor([1])

# Move the label tensor to the GPU if available
label = label.to(device, dtype=torch.float32)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
loss_fn = nn.MSELoss()

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Early stopping parameters
patience = 5
best_val_loss = float('inf')
epochs_without_improvement = 0

# Training loop
num_epochs = 40
for epoch in tqdm(range(num_epochs), desc='Epochs'):
    epoch_losses = [] # List to store the loss for each batch
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images).squeeze(1)
        loss = loss_fn(outputs, labels.squeeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item()) # Store the loss for each batch

    # Calculate and print the average loss for the epoch
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    logging.info(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    # Save the model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, f'models/model_checkpoint_epoch_{epoch + 1}.pt')

    # Validation loop
    avg_val_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).squeeze(1)
            loss = loss_fn(outputs, labels.squeeze(1))
            avg_val_loss += loss.item()
    avg_val_loss /= len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    logging.info(f'Validation Loss: {avg_val_loss:.4f}')

    # Adjust the learning rate
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
        }, 'models/best_model.pt')
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print('Early stopping triggered')
            logging.info('Early stopping triggered')
            break

# Testing loop
test_losses = []
with torch.no_grad(): # Disable gradient calculation
    model.eval()
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)
        outputs = model(images).squeeze(1)
        loss = loss_fn(outputs, labels.squeeze(1))
        test_losses.append(loss.item())

# Calculate and print the average loss for the test set
avg_test_loss = sum(test_losses) / len(test_losses)
print(f'Average Test Loss: {avg_test_loss:.4f}')
logging.info(f'Average Test Loss: {avg_test_loss:.4f}')