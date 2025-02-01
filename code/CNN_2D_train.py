import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import torchvision.models as models
from torch.utils.data import random_split
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from dataset_loader_2D import CustomMRIDataset
import logging
    
# Instantiate the dataset
dataset = CustomMRIDataset(csv_file='../tor/paths_labels.csv', img_dir='../tor/rel3_dhcp_anat_pipeline/')
img, label = dataset[0]
print(img.shape)
print(label)
torch.save(dataset, 'dataset.pt')

# Load the pre-trained model
resnet = models.resnet18()

# Change the last layer to output 1 class (regression)
resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7))
resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=1)

# Train/test/validation split
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create a label tensor
label = torch.tensor([1])

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assign the model and tensors to use GPU resources if available
resnet = resnet.to(device)
label = label.to(device, dtype=torch.float32)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
loss_fn = nn.MSELoss()

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Early stopping parameters
patience = 5
best_val_loss = float('inf')
epochs_without_improvement = 0

# Training loop
num_epochs = 10
for epoch in tqdm(range(num_epochs), desc='Epochs'):
    epoch_losses = [] # List to store the loss for each batch
    resnet.train()
    for i in range(0, len(train_dataset)):
        img, label = dataset[i]
        img = img.to(device, dtype=torch.float32).requires_grad_()
        label = label.to(device, dtype=torch.float32).requires_grad_()
        outputs = resnet(img)
        aggregated_output = outputs.mean(dim=0)
        loss = loss_fn(aggregated_output, label)
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
        'model_state_dict': resnet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, f'models/model_checkpoint_epoch_{epoch + 1}.pt')

    # Validation loop
    avg_val_loss = 0
    resnet.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_dataset):
            images = images.to(device)
            labels = labels.to(device)
            outputs = resnet(images)
            loss = loss_fn(outputs, labels)
            avg_val_loss += loss.item()
    avg_val_loss /= len(val_dataset)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    logging.info(f'Validation Loss: {avg_val_loss:.4f}')

    # Adjust the learning rate
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': resnet.state_dict(),
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
    resnet.eval()
    for i in range(0, len(test_dataset)):
        img, label = dataset[i]
        img = img.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.float32)
        outputs = resnet(img)
        loss = loss_fn(outputs, label)
        test_losses.append(loss.item())

# Calculate and print the average loss for the test set
avg_test_loss = sum(test_losses) / len(test_losses)
print(f'Average Test Loss: {avg_test_loss:.4f}')
logging.info(f'Average Test Loss: {avg_test_loss:.4f}')