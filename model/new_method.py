import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.optim.lr_scheduler import StepLR

# Constants
DATASET_PATH = 'GeneratedData/'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
NUM_FEATURES = 8
NUM_MODELS = 6  # Number of models in the ensemble 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_DECAY = 1e-4 # L2 Regularization parameter

# Utility Functions
def load_data_frame(filename, sep='\s+', header=None):
    filepath = os.path.join(DATASET_PATH, filename + '.txt')
    return pd.read_csv(filepath, header=header, sep=sep)

# Dataset class
class HAPTDataset(Dataset):
    def __init__(self, features, labels):
        total_features = features.shape[1]
        seq_len_options = [1]  # Possible sequence lengths
        seq_len = next((seq for seq in seq_len_options if total_features % seq == 0), 1)
        input_size = total_features // seq_len
        self.data = torch.tensor(features, dtype=torch.float32).view(-1, seq_len, input_size)
        self.labels = torch.tensor(labels.values - 1, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# GRU Network class
class GRUNetwork(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=64, num_layers=2):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 256)  # Multiply by 2 for bidirectional
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.batch_norm4 = nn.BatchNorm1d(32)

    def forward(self, x):
        x, _ = self.gru(x)  # x shape will be (batch_size, seq_len, hidden_size * 2)
        x = x[:, -1, :]  # x shape will be (batch_size, hidden_size * 2)
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.batch_norm3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm4(self.fc4(x)))
        x = self.fc5(x)
        return x

# Training function for one model
def train_single_model(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(y.view_as(pred)).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy

# Testing function for one model
def test_single_model(dataloader, model):
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            output = model(X)
            loss = loss_fn(output, y)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return accuracy, avg_loss, all_preds, all_labels

# Main Process
def main():
    # Load data
    train_set = load_data_frame('x_train')
    test_set = load_data_frame('x_test')
    
    # Load feature names
    with open(os.path.join(DATASET_PATH, 'feature_names.txt')) as f:
        features = [line.strip() for line in f]

    train_set.columns = features
    test_set.columns = features
    
    train_labels = load_data_frame('y_train', header=None)
    test_labels = load_data_frame('y_test', header=None)
    
    train_set['activity'] = train_labels
    test_set['activity'] = test_labels
    
    # Feature Selection using NCA
    X_train, X_val, y_train, y_val = train_test_split(train_set[features], train_set['activity'], test_size=0.2, random_state=42)
    
    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(test_set[features])
    
    # # NCA feature selection
    nca = NeighborhoodComponentsAnalysis(n_components=NUM_FEATURES, random_state=42)
    X_train_scaled = nca.fit_transform(X_train_scaled, y_train)
    X_val_scaled = nca.transform(X_val_scaled)
    X_test_scaled = nca.transform(X_test_scaled)
    important_features = np.abs(nca.components_).sum(axis=0)
    sorted_indices = np.argsort(important_features)[::-1]  # Sort in descending order

    # Get the feature names corresponding to the sorted indices
    selected_features = [features[i] for i in sorted_indices[:NUM_FEATURES]]
    print("Top selected features:", selected_features)

    # Prepare datasets
    train_dataset = HAPTDataset(X_train_scaled, y_train)
    val_dataset = HAPTDataset(X_val_scaled, y_val)
    test_dataset = HAPTDataset(X_test_scaled, test_set['activity'])

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create ensemble of models
    models = [GRUNetwork(input_size=NUM_FEATURES, num_classes=12).to(DEVICE) for _ in range(NUM_MODELS)]
    optimizers = [optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) for model in models]
    loss_function = nn.CrossEntropyLoss()

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    schedulers = [StepLR(optimizer, step_size=10, gamma=0.1) for optimizer in optimizers]

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        epoch_train_loss, epoch_train_accuracy = 0, 0
        epoch_val_loss, epoch_val_accuracy = 0, 0
        
        for model, optimizer,scheduler in zip(models, optimizers,schedulers):
            # Train the model
            train_loss, train_accuracy = train_single_model(train_dataloader, model, loss_function, optimizer)
            epoch_train_loss += train_loss
            epoch_train_accuracy += train_accuracy
            scheduler.step()

        # Evaluate on validation set
        for model in models:
            val_accuracy, val_loss, _, _ = test_single_model(val_dataloader, model)
            epoch_val_loss += val_loss
            epoch_val_accuracy += val_accuracy

        # Average out losses and accuracies
        avg_train_loss = epoch_train_loss / NUM_MODELS
        avg_train_accuracy = epoch_train_accuracy / NUM_MODELS
        avg_val_loss = epoch_val_loss / NUM_MODELS
        avg_val_accuracy = epoch_val_accuracy / NUM_MODELS

        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy * 100:.2f}%")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy * 100:.2f}%")

    # Evaluate ensemble models on test data
    test_accuracies = []
    all_preds = []
    all_labels = []
    test_losses = []

    for model in models:
        accuracy, loss, preds, labels = test_single_model(test_dataloader, model)
        test_accuracies.append(accuracy)
        all_preds.extend(preds)
        all_labels.extend(labels)
        test_losses.append(loss)

    avg_test_accuracy = np.mean(test_accuracies)
    avg_test_loss = np.mean(test_losses)
    print(f"\nTest Accuracy of Ensemble: {avg_test_accuracy * 100:.2f}%")
    print(f"Test Loss of Ensemble: {avg_test_loss:.2f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Normalize the confusion matrix to percentages
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

    # Plotting the normalized confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix (in %)')
    plt.show()
    # Plotting training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

    # Plotting training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, EPOCHS + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
