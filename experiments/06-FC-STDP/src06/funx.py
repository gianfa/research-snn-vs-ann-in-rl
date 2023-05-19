import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ML

def generate_data_batches(X, y, batch_size, shuffle=True):
    """
    
    Example
    -------
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    bs = 32
    train_loader, test_loader = generate_data_batches(X, y, bs)
    for batch_features, batch_labels in train_loader:
        print(batch_features.shape, batch_labels.shape)
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

