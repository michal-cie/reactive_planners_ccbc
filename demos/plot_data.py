import pickle
import numpy as np
import matplotlib.pyplot as plt

expert_data = 'data_X20_06_20_19_17_05'
expert_data_file_path = f"/home/michal/projects/contact_con/devel/reactive_planners/demos/learning_data/{expert_data}.pkl"

with open(expert_data_file_path, 'rb') as f:
        expert_data = pickle.load(f)

conditioning_type = 'des_vel'
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

des_vel_states  = np.array(expert_data['des_vel_states'])
contact_states  = np.array(expert_data['contact_states'])
actions         = np.array(expert_data['actions'])
pd_actions      = np.array(expert_data['pd_actions'])

seq = np.arange(des_vel_states.shape[0])

############################################################################
# Plots
############################################################################
# plt.figure()
# plt.plot(seq, des_vel_states[:,0:11], 'o', linestyle='none', markersize=1, label='q')
# plt.grid(True) 
# plt.legend()
# plt.figure()
# plt.plot(seq, des_vel_states[:,11:23], 'o', linestyle='none', markersize=1, label='qd')
# plt.grid(True) 
# plt.legend()
# plt.figure()
# plt.plot(seq, des_vel_states[:,23:25], 'o', linestyle='none', markersize=1, label='stance_phae')
# plt.legend()
# plt.grid(True) 
# plt.figure()
# plt.plot(seq, des_vel_states[:,25:], 'o', linestyle='none', markersize=1, label='des_velcond_var')
# plt.legend()
# plt.grid(True) 
plt.figure()
plt.plot(seq, contact_states[:,-1], 'o', linestyle='none', markersize=1, label='contact_cond_var')
plt.legend()
plt.grid(True)
# plt.figure()
# plt.plot(seq, actions, 'o', linestyle='none', markersize=1, label='contact_cond_var')
# plt.legend()
# plt.grid(True)
# plt.figure()
# plt.plot(seq, pd_actions, 'o', linestyle='none', markersize=1, label='contact_cond_var')
# plt.legend()
# plt.grid(True) 
plt.show()
