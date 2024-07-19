import os
import torch
import pickle
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from networks.networks2 import MLPPolicy, PolicyModel, normalize_columns, apply_normalization
from torch.utils.data import TensorDataset, DataLoader

script_path = os.path.abspath(__file__)
script_dir_path = os.path.dirname(script_path)

data_files = ['data_X69_06_24_12_17_50',
              'data_X132_06_24_12_09_11', 
              'data_X237_06_24_12_18_11', 
              'data_X513_06_24_12_34_14',
              'data_X939_06_24_15_07_42']

#for data_file, val_data_file in zip(data_files, val_data_files):
for conditioning_type in ['pd_des_vel']:
#for conditioning_type in ['pd_contact_gait', 'pd_contact', 'pd_des_vel']:
    # Load data
    expert_data_name = 'data_X100_06_26_16_00_18'
    expert_data_file_path = f"{script_dir_path}/learning_data/{expert_data_name}.pkl"
    expert_val_data = 'data_X50_06_26_15_48_59'
    expert_val_data_file_path = f"{script_dir_path}/learning_data/{expert_val_data}.pkl"

    #policy_dir = f'{script_dir_path}/{data_files[0]}/learned_policies'
    policy_dir = f'{script_dir_path}/learned_policies'
    
    with open(expert_data_file_path, 'rb') as f:
        expert_data = pickle.load(f)
    with open(expert_val_data_file_path, 'rb') as f:
        expert_val_data = pickle.load(f)

    #conditioning_type = 'des_vel'

    if conditioning_type == 'contact':
        states_key = 'contact_states'
        actions_key = 'actions'
    elif conditioning_type == 'pd_contact':
        states_key = 'contact_states'
        actions_key = 'pd_actions'
    elif conditioning_type == 'pd_two_contact':
        states_key = 'two_contact_states'
        actions_key = 'pd_actions'
    elif conditioning_type == 'pd_contact_gait':
        states_key = 'contact_gait_states'
        actions_key = 'pd_actions'
    elif conditioning_type == 'des_vel':
        states_key = 'des_vel_states'
        actions_key = 'actions'
    elif conditioning_type == 'pd_des_vel':
        states_key = 'des_vel_states'
        actions_key = 'pd_actions'

    data_states  = torch.tensor(np.array(expert_data[states_key]), dtype=torch.float32)[::1]
    data_actions = torch.tensor(np.array(expert_data[actions_key]), dtype=torch.float32)[::1]
    
    num_samples = data_states.size(0)
    indices = torch.randperm(num_samples)

    train_percent = 0.8
    split_idx = int(num_samples * train_percent)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    states = data_states[train_indices]
    actions = data_actions[train_indices]

    val_states = data_states[val_indices]
    val_actions = data_actions[val_indices]

    print(states.shape)
    print(actions.shape)
    print(val_states.shape)
    print(val_actions.shape)

    train_dataset = TensorDataset(states, actions)
    val_dataset = TensorDataset(val_states, val_actions)

    # Creating DataLoaders for the training and validation sets
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Hyperparameters
    input_size = states.shape[1]
    hidden_size = 512
    output_size = actions.shape[1]
    layers = 3
    dropout_prob = 0.0
    learning_rate = 0.0001
    num_epochs = 26

    # Initialize policy
    policy = MLPPolicy(input_size, hidden_size, output_size, layers, dropout_prob=dropout_prob)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Best model initialization
    best_val_loss = float('inf')
    best_policy = None

    train_losses = []
    val_losses = []

    try:
        for epoch in range(num_epochs):
            # Training phase
            policy.train()
            total_train_loss = 0
            total_samples = 0
            for batch_states, batch_actions in train_loader:
                optimizer.zero_grad()
                predicted_train_actions = policy(batch_states)
                train_loss = criterion(predicted_train_actions, batch_actions)
                train_loss.backward()
                optimizer.step()
                total_train_loss += train_loss.item()
            average_train_loss = total_train_loss / len(train_loader)

            # Validation phase
            policy.eval()
            total_val_loss = 0
            total_samples = 0
            with torch.no_grad():
                for batch_states, batch_actions in val_loader:
                    predicted_val_actions = policy(batch_states)
                    val_loss = criterion(predicted_val_actions, batch_actions)
                    total_val_loss += val_loss.item()
            average_val_loss = total_val_loss / len(val_loader)

            train_losses.append(average_train_loss)
            val_losses.append(average_val_loss)

            # Save best model
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                best_policy = policy

            print(f'Epoch {epoch}, Train Loss: {average_train_loss:.4e}  Val Loss: {average_val_loss:.4e}  Best Val Loss: {best_val_loss:.4e}')
            if epoch % 5 == 0: #and epoch != 0:
                policy_model = PolicyModel(best_policy)
                policy_model = PolicyModel(policy)
                actions_str = 'pd_' if 'pd' in actions_key else ''
                policy_model.save_model(policy_dir, f'{actions_str}{states_key}_ep{epoch}.pth')
                policy_model.save_model(policy_dir, f'train_{actions_str}{states_key}_ep{epoch}.pth')
                print(f"Best validation loss: {best_val_loss}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving last best model.")

    losses = {'train_losses': [],
              'val_losses': []}
    losses['train_losses'] = train_losses
    losses['val_losses'] = val_losses
    losses_path = f"{policy_dir}/losses/{conditioning_type}_{expert_data_name}.pkl"
    # with open(losses_path, 'wb') as f:
    #     pickle.dump(losses, f)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Contact Conditioned Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.show()