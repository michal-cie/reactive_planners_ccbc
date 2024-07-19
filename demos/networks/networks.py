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
    def __init__(self, model, state_mean, state_std, action_mean, action_std):
        """
        Initialize the PolicyModel.

        :param model: The policy model (a PyTorch model).
        :param state_mean: Mean of the states for normalization.
        :param state_std: Standard deviation of the states for normalization.
        :param action_mean: Mean of the actions for denormalization.
        :param action_std: Standard deviation of the actions for denormalization.
        """
        self.model = model
        self.state_mean = state_mean.clone().detach()
        self.state_std = state_std.clone().detach()
        self.action_mean = action_mean.clone().detach()
        self.action_std = action_std.clone().detach()

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
            'state_mean': self.state_mean,
            'state_std': self.state_std,
            'action_mean': self.action_mean,
            'action_std': self.action_std
        }, model_path)

    def load_model(self, model_path):
        """
        Load a model and its statistics from a file.

        :param model_path: Path to the model file.
        """
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.state_mean = checkpoint['state_mean']
        self.state_std = checkpoint['state_std']
        self.action_mean = checkpoint['action_mean']
        self.action_std = checkpoint['action_std']

    def infer(self, states):
        """
        Perform inference with normalization and denormalization.

        :param states: Input states as a torch tensor.
        :return: Inferred actions as a torch tensor.
        """
        # Normalize the states
        normalized_states = (states - self.state_mean) / self.state_std

        # Perform inference
        self.model.eval()
        with torch.no_grad():
            normalized_actions = self.model(normalized_states)

        # Denormalize the actions
        actions = normalized_actions * self.action_std + self.action_mean

        return actions