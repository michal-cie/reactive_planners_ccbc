import os
import pickle
import numpy as np

script_path = os.path.abspath(__file__)
script_dir_path = os.path.dirname(script_path)

data_files = ['data_X30_06_24_12_21_16',
              'data_X69_06_24_12_17_50',
              'data_X132_06_24_12_09_11', 
              'data_X237_06_24_12_18_11', 
              'data_X513_06_24_12_34_14',
              'data_X939_06_24_15_07_42']

for data in data_files:
    expert_data_name = data
    expert_data_file_path = f"{script_dir_path}/learning_data/{expert_data_name}.pkl"
    new_expert_data_file_path = f"{script_dir_path}/learning_data/modified_data/{expert_data_name}.pkl"

    with open(expert_data_file_path, 'rb') as f:
        expert_data = pickle.load(f)

    expert_data['contact_des_vel_states'] = []
    expert_data['contact_gait_des_vel_states'] = []

    for i in range(len(expert_data['des_vel_states'])):
        expert_data['contact_des_vel_states'].append(expert_data['contact_states'][i] + [expert_data['des_vel_states'][i][-3]])
        expert_data['contact_gait_des_vel_states'].append(expert_data['contact_gait_states'][i] + [expert_data['des_vel_states'][i][-3]])

    print('Contact states:', np.array(expert_data['contact_states']).shape)
    print('Contact Des Vel states:', np.array(expert_data['contact_des_vel_states']).shape)

    print('Contact Gait states:', np.array(expert_data['contact_gait_states']).shape)
    print('Contact Gait Des Vel states:', np.array(expert_data['contact_gait_des_vel_states']).shape)

    with open(new_expert_data_file_path, 'wb') as f:
        pickle.dump(expert_data, f)
