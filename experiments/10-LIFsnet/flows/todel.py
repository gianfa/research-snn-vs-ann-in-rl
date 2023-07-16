""""

------------



"""
# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Parametri del segnale sinusoidale
amplitude = 1.0
frequency = 0.1
phase = 0.0

# Parametri delle anomalie
num_anomalies = 10
anomaly_magnitude = 5.0

# Parametri di addestramento
num_epochs = 100
batch_size = 64

def generate_signal(length):
    time = np.arange(length)
    signal = amplitude * np.sin(2 * np.pi * frequency * time + phase)
    return signal

def generate_anomalies(length):
    anomalies = np.zeros(length)
    anomaly_indices = np.random.choice(length, num_anomalies, replace=False)
    anomalies[anomaly_indices] = anomaly_magnitude
    return anomalies

def create_examples(length):
    signal = generate_signal(length)
    anomalies = generate_anomalies(length)
    X = signal + anomalies
    y = (anomalies != 0).astype(int)
    return X, y

def check_data_dimensions(data_loader):
    x_shape = None
    y_shape = None
    for batch_X, batch_y in data_loader:
        # Controlla le dimensioni di X
        if x_shape is None:
            x_shape = batch_X.shape[1:]
        else:
            if batch_X.shape[1:] != x_shape:
                raise Exception("Le dimensioni di X non sono coerenti all'interno del DataLoader")
        
        # Controlla le dimensioni di y
        if y_shape is None:
            y_shape = batch_y.shape[1:]
        else:
            if batch_y.shape[1:] != y_shape:
                raise Exception("Le dimensioni di y non sono coerenti all'interno del DataLoader")
    
    return True, "Le dimensioni di X e y sono coerenti"

# Divisione del dataset in train, validation e test set
def train_valid_test_split(X, y, valid_ratio=0.15, test_ratio=0.2):
    dataset_length = len(X)
    test_length = int(dataset_length * test_ratio)
    valid_length = int(dataset_length * valid_ratio)
    train_length = dataset_length - test_length - valid_length

    train_X, train_y = X[:train_length], y[:train_length]
    valid_X, valid_y = X[train_length:train_length + valid_length], y[train_length:train_length + valid_length]
    test_X, test_y = X[train_length + valid_length:], y[train_length + valid_length:]

    return train_X, train_y, valid_X, valid_y, test_X, test_y

# Modello di anomaly detection basato su LSTM
class LSTMAnomalyDetection(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMAnomalyDetection, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _= self.lstm(x)
        out = self.fc(x).squeeze()
        return out

# Addestramento del modello
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).unsqueeze(1)
        
        # Controllo delle dimensioni
        if labels.size() != outputs.size():
            raise ValueError(f"Target size {labels.size()} must be the same as input size {outputs.size()}")
        
        loss = criterion(torch.sigmoid(outputs), labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Valutazione delle performance del modello
def evaluate_model(model, data_loader):
    model.eval()
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            predicted = torch.round(outputs.squeeze())
            true_positives += torch.sum(predicted * labels).item()
            false_positives += torch.sum(predicted * (1 - labels)).item()
            false_negatives += torch.sum((1 - predicted) * labels).item()
    precision = true_positives / (true_positives + false_positives + 1e-5)
    recall = true_positives / (true_positives + false_negatives + 1e-5)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-5)
    return precision, recall, f1_score

# Create the dataset
dataset_length = 1000
X, y = create_examples(dataset_length)

# Split the dataset
train_X, train_y, valid_X, valid_y, test_X, test_y = train_valid_test_split(X, y)

train_X_tensor = torch.from_numpy(train_X).unsqueeze(1).float()
train_y_tensor = torch.from_numpy(train_y).unsqueeze(1).float()
valid_X_tensor = torch.from_numpy(valid_X).unsqueeze(1).float()
valid_y_tensor = torch.from_numpy(valid_y).unsqueeze(1).float()
test_X_tensor = torch.from_numpy(test_X).unsqueeze(1).float()
test_y_tensor = torch.from_numpy(test_y).unsqueeze(1).float()

# convert all to DataLoaders
train_dataset = torch.utils.data.TensorDataset(train_X_tensor, train_y_tensor)
valid_dataset = torch.utils.data.TensorDataset(valid_X_tensor, valid_y_tensor)
test_dataset = torch.utils.data.TensorDataset(test_X_tensor, test_y_tensor)


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True)

result, message = check_data_dimensions(train_loader)
result, message = check_data_dimensions(valid_loader)
result, message = check_data_dimensions(test_loader)

# %%

# Model Definition
input_size = 1
hidden_size = 16
output_size = 1
model = LSTMAnomalyDetection(input_size, hidden_size, output_size)

# Definizione della funzione di loss e dell'ottimizzatore
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# Addestramento del modello
train_losses = []
valid_losses = []
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer)
    valid_loss = evaluate_model(model, valid_loader)[2]
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f} | Valid F1-Score: {valid_loss:.4f}")

# Valutazione delle performance finali sul test set
precision, recall, f1_score = evaluate_model(model, test_loader)
print(f"\nTest Precision: {precision:.4f} | Test Recall: {recall:.4f} | Test F1-Score: {f1_score:.4f}")

# Visualizzazione dei grafici
def plot_signal(signal):
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title("Segnale originale")
    plt.xlabel("Tempo")
    plt.ylabel("Ampiezza")
    plt.show()

def plot_examples(X, y, predictions=None):
    plt.figure(figsize=(10, 4))
    plt.plot(X, label="Segnale affetto da anomalie")
    plt.plot(y, "r", label="Labels")
    if predictions is not None:
        anomaly_indices = np.where(predictions == 1)[0]
        plt.scatter(anomaly_indices, X[anomaly_indices], color="g", label="Anomalie predette")
    plt.title("Esempi di segnali e relative labels")
    plt.xlabel("Tempo")
    plt.ylabel("Ampiezza")
    plt.legend()
    plt.show()

def plot_performance(train_losses, valid_losses):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Valid F1-Score")
    plt.title("Performance del modello")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/F1-Score")
    plt.legend()
    plt.show()

# Plot del segnale originale
plot_signal(X)

# Plot di alcuni esempi X e relative labels y
num_examples = 5
indices = np.random.choice(len(test_X), num_examples, replace=False)
sample_X = test_X[indices]
sample_y = test_y[indices]
predictions = torch.round(model(torch.from_numpy(sample_X).unsqueeze(1).float())).numpy()
plot_examples(sample_X, sample_y, predictions)

# Plot del confronto tra labels e predizioni
all_predictions = torch.round(model(test_X_tensor)).numpy()
plot_examples(test_X, test_y, all_predictions)

# Plot delle metriche di performance
plot_performance(train_losses, valid_losses)

# %%
