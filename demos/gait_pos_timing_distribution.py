import os
import torch
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt

script_path = os.path.abspath(__file__)
script_dir_path = os.path.dirname(script_path)

data_files = ['data_X100_06_27_12_10_37',
              'data_X50_06_27_11_59_24', 
              'data_X25_06_27_11_50_40', 
              'data_X200_06_27_12_52_10',
              'data_X400_06_27_15_23_05']

expert_data_name = 'data_X25_06_27_11_50_40'
expert_data_file_path = f"{script_dir_path}/learning_data/{expert_data_name}.pkl"

policy_dir = f'{script_dir_path}/learned_policies'

with open(expert_data_file_path, 'rb') as f:
    expert_data = pickle.load(f)

conditioning_type = 'pd_contact_gait'

if conditioning_type == 'contact':
    states_key = 'contact_states'
    actions_key = 'actions'
elif conditioning_type == 'pd_contact':
    states_key = 'contact_states'
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

# gait_idx = [-1] if gait == 'walk' else [1]
mask = data_states[:, -2] == -1 

# Splitting the tensor into two based on the mask
walk_states = data_states[mask]
run_states = data_states[~mask]

ds= 10
plt.figure()
plt.scatter(walk_states[::ds,-5], walk_states[::ds,-1], label='Walk', alpha=0.5, s=1)
plt.scatter(run_states[::ds,-5], run_states[::ds,-1], label='Run', alpha=0.5, s=1)
plt.xlabel('x pos')
plt.ylabel('timing')
plt.legend()
plt.grid(True) 
plt.figure()
plt.scatter(walk_states[::ds,-5], walk_states[::ds,-1], label='Walk', alpha=0.5, s=1)
plt.xlabel('x pos')
plt.ylabel('timing')
plt.legend()
plt.grid(True) 
# Specify that this will be a 3D plot
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(walk_states[::ds, -5], walk_states[::ds, -4], walk_states[::ds, -1], label='Walk', color='yellow', alpha=0.5, s=1)
ax.scatter(run_states[::ds, -5], run_states[::ds, -4], run_states[::ds, -1], label='Run', color='blue', alpha=0.5, s=1)
ax.set_xlabel("x pos")
ax.set_ylabel("y pos")
ax.set_zlabel("timing")
ax.legend()
# Adding grid
ax.grid(True)
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(walk_states[::ds, 0], walk_states[::ds, -4], walk_states[::ds, -1], label='Walk', color='yellow', alpha=0.5, s=1)
ax.scatter(run_states[::ds, 0], run_states[::ds, 13], run_states[::ds, -1], label='Run', color='blue', alpha=0.5, s=1)
ax.set_xlabel("z pos")
ax.set_ylabel("z_vel")
ax.set_zlabel("timing")
ax.legend()
# Adding grid
ax.grid(True)
plt.show()