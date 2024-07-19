import torch
import numpy as np
from matplotlib import pyplot as plt
from networks.networks2 import PolicyModel
from walk_run_batch import walk_run_batch

if __name__ == '__main__':

    policy = 'pd_contact_states_ep200'
    policy_dir = '/home/michal/projects/contact_con/devel/reactive_planners/demos/learned_policies'
    dataset_dir = 'data_X400_07_10_14_42_59'
    policy_file_path = f"{policy_dir}/{dataset_dir}/{policy}.pth"
    # policy_file_path = f"{policy_dir}/{policy}.pth"
    trained_policy = torch.load(policy_file_path)
    trained_policy = PolicyModel(trained_policy['model'])

    collect_learning_data = False
    visualize_pybullet = True
    conditioning_type = "pd_two_contact" # Options: "contact", "des_vel"
    max_dv = 1.5#1.5
    min_vel = -1.0#-1.0
    max_vel = 1.3#1.3
    min_gait_time = 1#1
    max_gait_time = 3#3
    min_com_height = 0.30#0.29
    max_com_height = 0.30#0.31
    rand_init_cond = False
    force_disturbance = False

    total_learning_data = None
    total_data = None

    num_batch_goals = 3
    num_batches = 1
    for _ in range(num_batches):
        batch_data, batch_learning_data = \
        walk_run_batch(max_dv, min_vel, max_vel, 
                       min_gait_time, max_gait_time, 
                       min_com_height, max_com_height,
                       num_batch_goals,
                       collect_learning_data = collect_learning_data,
                       learned_policy = trained_policy,
                       conditioning_type = conditioning_type,
                       rand_init_cond = rand_init_cond,
                       force_disturbance = force_disturbance,
                       visualize_pybullet = visualize_pybullet)
        if total_learning_data is None:
            total_learning_data = batch_learning_data
            total_data = batch_data
            continue
        for key in total_learning_data:
            total_learning_data[key].extend(batch_learning_data[key])
        for key in total_data:
            total_data[key].extend(batch_data[key])

############################################################################
# Plots
############################################################################
plt.figure()
plt.plot(total_data['time'], np.array(total_data['actual_com_velocity'])[:,0], label='CoM_x_dot')
plt.plot(total_data['time'], np.array(total_data['desired_com_velocity'])[:,0], label='Target Vel')
plt.legend()
plt.grid(True) 
plt.figure()
plt.plot(np.array(total_data['current_desired_contact'])[:,0], label='des_contact x')
plt.plot(np.array(total_data['actual_contact'])[:,0], label='act_contact x')
plt.legend()
plt.grid(True)
plt.figure()
plt.plot(np.array(total_data['current_desired_contact'])[:,1], label='des_contact y')
plt.plot(np.array(total_data['actual_contact'])[:,1], label='act_contact y')
plt.legend()
plt.grid(True) 
plt.show()