import os
import datetime
import torch
import torch.nn as nn

class MLPPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers, dropout_prob=0.5):
        super(MLPPolicy, self).__init__()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_prob))

        # Additional hidden layers
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_prob))

        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class PolicyModel:
    def __init__(self, model):
        """
        Initialize the PolicyModel.

        :param model: The policy model (a PyTorch model).
        """
        self.model = model

    def save_model(self, directory, model_name=None):
        """
        Save the model and its statistics.

        :param directory: Directory where to save the model.
        :param model_name: Optional custom name for the model file.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        if model_name is None:
            current_datetime = datetime.datetime.now()
            datetime_str = current_datetime.strftime("%m_%d_%H_%M_%S")
            model_name = f"policy_model_{datetime_str}.pth"

        model_path = os.path.join(directory, model_name)

        torch.save({
            'model': self.model,
        }, model_path)

    def load_model(self, model_path):
        """
        Load a model and its statistics from a file.

        :param model_path: Path to the model file.
        """
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
    def infer(self, states):
        """
        Perform inference.

        :param states: Input states as a torch tensor.
        :return: Inferred actions as a torch tensor.
        """
        self.model.eval()
        
        with torch.no_grad():
            actions = self.model(states)

        return actions
    
def normalize_columns(data, column_indices):
    data_normalized = data.clone()
    data_stats = []

    # Normalize each specified column and collect details
    for idx in column_indices:
        column = data[:, idx]
        mean = torch.mean(column)
        std = torch.std(column) + 1e-6  # Avoid division by zero
        data_normalized[:, idx] = (column - mean) / std

        # Append normalization details for the current column
        data_stats.append({
            'index': idx,
            'mean': mean.item(),
            'std': std.item()
        })

    return data_normalized, data_stats

def apply_normalization(data, normalization_stats):

    data_normalized = data.clone()

    # Apply normalization for each specified column using the provided details
    for detail in normalization_stats:
        idx = detail['index']
        mean = detail['mean']
        std = detail['std']
        
        # Ensure the column index is within the new data's shape
        if idx < data.shape[1]:
            data_normalized[:, idx] = (data[:, idx] - mean) / std

    return data_normalized

def apply_denormalization(data, normalization_stats):

    data_denormalized = data.clone()

    # Apply normalization for each specified column using the provided details
    for detail in normalization_stats:
        idx = detail['index']
        mean = detail['mean']
        std = detail['std']
        
        # Ensure the column index is within the new data's shape
        if idx < data.shape[1]:
            data_denormalized[:, idx] = data[:, idx] * std + mean

    return data_denormalized