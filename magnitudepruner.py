import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import wandb
wandb.init(project="project_Newl5", entity="Gousia")
from torchsummary import summary
# Define constants
IMAGE_SIZE = 256
BATCH_SIZE = 256

torch.autograd.set_detect_anomaly(True)
print(f"PyTorch version: {torch.__version__}")

# Import metrics from sklearn
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report

# Define a custom dataset class
class EyeGazeDataset(Dataset):
    """
    Custom dataset class for the Eye Gaze dataset.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Define transformations
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Load and preprocess data
df = pd.read_csv('Eye_Gaze.csv')
data = df.replace({
    'eyegaze': {
        'away from camera': 0, 'looking at camera': 10, 'almost looking at camera': 5
    },
})

# Group the data by image path and calculate the mean 'eyegaze' value
result_df = data.groupby('image_path')['eyegaze'].mean().reset_index()
X = result_df['image_path'].values
Y = result_df['eyegaze'].values.astype(np.float32)

# Add 'Public/' prefix to image paths
for i in range(len(X)):
    X[i] = 'Public/' + X[i].split('=')[1]

# Print unique values and their counts in Y
p = np.unique(Y, return_counts=True)
for i in range(len(p[0])):
    print(f"{p[0][i]} ---------- {p[1][i]}")

# Split data
train_x, val_x, train_y, val_y = train_test_split(X, Y, test_size=0.1, random_state=32, shuffle=True)

num_train = len(train_x)
num_val = len(val_x)

print(f'Number of train images: {num_train}')
print(f'Number of val images: {num_val}')

# Create datasets
train_dataset = EyeGazeDataset(train_x, train_y, transform=transform)
val_dataset = EyeGazeDataset(val_x, val_y, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
else:
    print("CUDA is not available.")
    device = torch.device("cpu")

print(f"Number of CUDA devices available: {torch.cuda.device_count()}")

class CustomModel(nn.Module):
    """
    Custom model architecture for the Eye Gaze dataset.
    """
    def __init__(self):
        super(CustomModel, self).__init__()
        self.pretrained_model_mobilenet = models.mobilenet_v3_large(pretrained=True).to(device)
        self.pretrained_model_mobilenet.classifier = nn.Identity()

        for param in self.pretrained_model_mobilenet.parameters():
            param.requires_grad = True

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(960, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pretrained_model_mobilenet(x)
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

# Define a custom loss function
class Custom_CE_Loss(nn.Module):
    """
    Custom loss function for the Eye Gaze dataset.
    """
    def __init__(self):
        super(Custom_CE_Loss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, y_true, y_pred):
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        y_true = y_true.view(-1,)
        y_pred = y_pred.view(-1)

        # Filter values where |y_true - y_pred| > 2.5
        filtered_idx = torch.abs(y_true_flat - y_pred_flat) > 2.5
        y_true_filtered = y_true_flat[filtered_idx]
        y_pred_filtered = y_pred_flat[filtered_idx]

        # Calculate MSE loss for filtered values
        mse = self.l1_loss(y_true_filtered, y_pred_filtered)

        # Calculate MSE loss for all values
        mse1 = self.l1_loss(y_true, y_pred)

        # Calculate final loss
        loss = mse1 * (mse.item() / mse1.item())

        return loss



#Pruning Model using Dependency graph
import torch
import torch_pruning as tp
import warnings
warnings.filterwarnings('ignore')
import sys, os
sys.path.append(os.path.abspath("../"))

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained model from a saved state dictionary
model = CustomModel()
state_dict = torch.load('saved_model.pth')
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Example inputs for the model
import torch

# Define your tensor
example_inputs = torch.randn(1, 3, 256, 256)

# Define the device you want to move the tensor to
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the tensor to the specified device
example_inputs = example_inputs.to(device)
# 0. importance criterion for parameter selections
imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')

# 1. ignore some layers that should not be pruned, e.g., the final classifier layer.
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1:
        ignored_layers.append(m) # DO NOT prune the final classifier!

        
# 2. Pruner initialization
iterative_steps = 5 # You can prune your model to the target pruning ratio iteratively.
pruner = tp.pruner.MagnitudePruner(
    model, 
    example_inputs, 
    global_pruning=False, # If False, a uniform ratio will be assigned to different layers.
    importance=imp, # importance criterion for parameter selection
    iterative_steps=iterative_steps, # the number of iterations to achieve target ratio
    pruning_ratio=0.7, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256  This method removes weights with small magnitude in the network, resulting in a smaller and faster model, without too much performance lost in accuracy}
    ignored_layers=ignored_layers,
)

base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
for i in range(iterative_steps):
    # 3. the pruner.step will remove some channels from the model with least importance
    pruner.step()
    
    # 4. Do whatever you like here, such as fintuning
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(model)
    print(model(example_inputs).shape)
    print(
        "  Iter %d/%d, Params: %.2f M => %.2f M"
        % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6)
    )
    print(
        "  Iter %d/%d, MACs: %.2f G => %.2f G"
        % (i+1, iterative_steps, base_macs / 1e9, macs / 1e9)
    )
# Test forward pass
output = model(example_inputs)
print("Output.shape: ", output.shape)


# Saving the pruned model
file_path = 'M0.7.pth'
pruned_state_dict = tp.state_dict(model)  # Get the state dictionary from the pruned model
torch.save(pruned_state_dict, file_path)
print(f"Pruned model successfully saved at {file_path}")

# Loading the pruned model
print("Loading pruned model")
pruned_state_dict = torch.load('M0.7.pth')
tp.load_state_dict(model, state_dict=pruned_state_dict)
print("After pruning:")
model_summary = summary(model, input_size=(3, 256, 256))

# Retrain or fine-tune the pruned model
import torch
import random
import numpy as np
seed = 43
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
weight_decay = 5e-4
criterion = Custom_CE_Loss()

# Define optimizer and scheduler for the pruned model (model1)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=1, min_lr=1e-9)
mae_metric = nn.L1Loss()
mse_metric = nn.MSELoss()
print('iamgood')
num_epochs = 10
model.train()
loss_met = AverageMeter()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    mae = 0.0
    mse = 0.0
    total_samples = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        total_samples += inputs.size(0)

        # Metrics
        mae += nn.L1Loss()(outputs, targets).item()
        mse += nn.MSELoss()(outputs, targets).item()

    # Compute average loss and metrics for the epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_mae = mae / total_samples
    epoch_mse = mse / total_samples

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_mae = 0.0
    val_mse = 0.0
    val_total_samples = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)  # Use model1 for validation
            loss = criterion(outputs, targets)
            val_running_loss += loss.item()
            val_total_samples += inputs.size(0)

            # Metrics
            val_mae += nn.L1Loss()(outputs, targets).item()
            val_mse += nn.MSELoss()(outputs, targets).item()

    # Compute average validation loss and metrics
    val_loss = val_running_loss / len(val_loader)
    val_mae = val_mae / val_total_samples
    val_mse = val_mse / val_total_samples

    wandb.log({"loss": epoch_loss, "val_loss": val_loss, "MAE": epoch_mae, "val_MAE": val_mae, "MSE": epoch_mse, "val_MSE": val_mse})
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, MAE: {epoch_mae:.4f}, Val MAE: {val_mae:.4f}, MSE: {epoch_mse:.4f}, Val MSE: {val_mse:.4f}")

    # Update learning rate based on validation loss
    scheduler.step(val_loss)

# Save the fine-tuned model
file_path = 'Mf0.7.pth'
torch.save(model.state_dict(), file_path)
print(f"Fine-tuned model successfully saved at {file_path}")

# Load the saved fine-tuned model
model.load_state_dict(torch.load(file_path))
print(f"Fine-tuned model successfully loaded from {file_path}")
model.eval()
model_summary = summary(model, input_size=(3, 256, 256))
print(model_summary) 





