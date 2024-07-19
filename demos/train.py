import torch
import pickle
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from networks.networks import MLPPolicy, PolicyModel
from torch.utils.data import TensorDataset, DataLoader

for conditioning_type in ['des_vel']:
#for conditioning_type in ['contact', 'pd_contact', 'des_vel', 'pd_des_vel']:
    # Load data
    unified_expert_data = 'data_X43_06_17_23_07_43'
    unified_expert_data_file_path = f"/home/michal/projects/contact_con/devel/reactive_planners/demos/learning_data/{unified_expert_data}.pkl"

    unified_val_data = 'data_X20_06_17_22_56_30'
    unified_val_data_file_path = f"/home/michal/projects/contact_con/devel/reactive_planners/demos/learning_data/{unified_val_data}.pkl"

    with open(unified_expert_data_file_path, 'rb') as f:
        unified_expert_data = pickle.load(f)

    with open(unified_val_data_file_path, 'rb') as f:
        unified_val_data = pickle.load(f)

    policy_dir = '/home/michal/projects/contact_con/devel/reactive_planners/demos/learned_policies'
    #conditioning_type = 'des_vel'

    if conditioning_type == 'contact':
        states_key = 'contact_states'
        actions_key = 'actions'
    elif conditioning_type == 'pd_contact':
        states_key = 'contact_states'
        actions_key = 'pd_actions'
    elif conditioning_type == 'des_vel':
        states_key = 'des_vel_states'
        actions_key = 'actions'
    elif conditioning_type == 'pd_des_vel':
        states_key = 'des_vel_states'
        actions_key = 'pd_actions'

    states      = torch.tensor(np.array(unified_expert_data[states_key]), dtype=torch.float32)[::1]
    actions     = torch.tensor(np.array(unified_expert_data[actions_key]), dtype=torch.float32)[::1]
    val_states  = torch.tensor(np.array(unified_val_data[states_key]), dtype=torch.float32)[::1]
    val_actions = torch.tensor(np.array(unified_val_data[actions_key]), dtype=torch.float32)[::1]

    # Normalizing the data
    state_mean = (torch.mean(states, dim=0)).clone().detach()
    state_std = (torch.std(states, dim=0) + 1e-6).clone().detach()
    action_mean = (torch.mean(actions, dim=0)).clone().detach()
    action_std = (torch.std(actions, dim=0) + 1e-6).clone().detach()

    states = (states - state_mean) / state_std
    actions = (actions - action_mean) / action_std
    
    # noise_std = 0.01  
    # gaussian_noise = torch.randn(states.size()) * noise_std
    # states = states + gaussian_noise

    train_dataset = TensorDataset(states, actions)
    val_dataset = TensorDataset(val_states, val_actions)

    # Create DataLoaders
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Hyperparameters
    input_size = states.shape[1]
    hidden_size = 512
    output_size = actions.shape[1]
    layers = 4
    dropout_prob = 0.2
    learning_rate = 0.0001
    num_epochs = 101

    # Initialize policy
    policy = MLPPolicy(input_size, hidden_size, output_size, layers, dropout_prob=dropout_prob)
    criterion = nn.L1Loss()
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
                    batch_states = (batch_states - state_mean) / state_std
                    batch_actions = (batch_actions - action_mean) / action_std
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

            print(f'Epoch {epoch}, Train Loss: {train_loss.item():.4e}  Val Loss: {average_val_loss:.4e}  Best Val Loss: {best_val_loss:.4e}')
            if epoch % 1 == 0: #and epoch != 0:
                policy_model = PolicyModel(best_policy, state_mean, state_std, action_mean, action_std)
                actions_str = 'pd_' if 'pd' in actions_key else ''
                policy_model.save_model(policy_dir, f'{actions_str}{states_key}_ep{epoch}.pth')
                print(f"Best validation loss: {best_val_loss}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving last best model.")

# Save trained 
# current_datetime = datetime.datetime.now()
# datetime_str = current_datetime.strftime("%m_%d_%H_%M_%S")
# policy_model = PolicyModel(best_policy, state_mean, state_std, action_mean, action_std)
# policy_model.save_model('/home/michal/thesis/CCBC/trained_policies', f'{states_key}_ep{epoch}_{datetime_str}.pth')
# print(f"Best validation loss: {best_val_loss}")

# Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.title('Contact Conditioned Training and Validation Losses')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.yscale('log')
# plt.legend()
# plt.show()