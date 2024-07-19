import os
import torch
import pickle
import datetime
import numpy as np

script_path = os.path.abspath(__file__)
script_dir_path = os.path.dirname(script_path)

data_files = ['data_X200_06_27_12_52_10',
              'data_X200_06_27_14_13_36']

data_file1 = 'data_X200_06_27_12_52_10'
data_file_path1 = f"{script_dir_path}/learning_data/{data_file1}.pkl"
data_file2 = 'data_X200_06_27_14_13_36'
data_file_path2 = f"{script_dir_path}/learning_data/{data_file2}.pkl"

with open(data_file_path1, 'rb') as f:
    data_file1 = pickle.load(f)
with open(data_file_path2, 'rb') as f:
    data_file2 = pickle.load(f)

for key in data_file1:
    data_file1[key].extend(data_file2[key])

# Save data
current_datetime = datetime.datetime.now()
datetime_str = current_datetime.strftime("%m_%d_%H_%M_%S")
file_path = f"/home/michal/projects/contact_con/devel/reactive_planners/demos/learning_data/data_X{str(400)}_{datetime_str}.pkl"

data_states  = torch.tensor(np.array(data_file1['des_vel_states']), dtype=torch.float32)[::1]
data_actions = torch.tensor(np.array(data_file1['pd_actions']), dtype=torch.float32)[::1]
print(data_states.shape())
print(data_actions.shape())
exit()

with open(file_path, 'wb') as f:
    pickle.dump(data_file1, f)