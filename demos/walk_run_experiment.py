import numpy as np
from matplotlib import pyplot as plt
from walk_run_batch import walk_run_batch

if __name__ == '__main__':

    collect_learning_data = True
    visualize_pybullet = True
    conditioning_type = "contact" # Options: "contact", "des_vel"
    max_dv = 1.5
    min_vel = -1.0
    max_vel = 1.3
    min_gait_time = 1
    max_gait_time = 3
    min_com_height = 0.29
    max_com_height = 0.31
    rand_init_cond = True

    total_learning_data = {'contact_states': [],
                           'des_vel_states': [],
                           'actions': [],
                           'pd_actions': []}

    total_data = {
        'time': [],
        'desired_com_velocity': [],
        'actual_com_velocity': [],
        'next_desired_contact': [],
        'current_desired_contact': [],
        'actual_contact': [],
        'actual_com_position': [],
        'fails': [],
    }

    num_batch_goals = 1
    num_batches = 1
    for _ in range(num_batches):
        batch_data, batch_learning_data = \
        walk_run_batch(max_dv, min_vel, max_vel, 
                       min_gait_time, max_gait_time, 
                       min_com_height, max_com_height,
                       num_batch_goals,
                       collect_learning_data = collect_learning_data,
                       learned_policy=None,
                       conditioning_type = conditioning_type,
                       rand_init_cond = rand_init_cond,
                       visualize_pybullet = visualize_pybullet)
    
        for key in total_learning_data:
            total_learning_data[key].extend(batch_learning_data[key])
        for key in total_data:
            total_data[key].extend(batch_data[key])
    
    for i in range(num_batches):
        print(f"Batch {i}")
        print(f"Desired com velocity: {total_data['desired_com_velocity'][i][-5:]}")
        print(f"Actual com velocity: {total_data['actual_com_velocity'][i][-5:]}")
        print(f"Next desired contact: {total_data['next_desired_contact'][i][-5:]}")
        print(f"Current desired contact: {total_data['current_desired_contact'][i][-5:]}")
        print(f"Actual contact: {total_data['actual_contact'][i][-5:]}")
        print(f"Actual com position: {total_data['actual_com_position'][i][-5:]}")
        print(f"Fails: {total_data['fails'][i][-5:]}")
        print(f"Contact states: {total_learning_data['contact_states'][i][-5:]}")
        print(f"Des vel states: {total_learning_data['des_vel_states'][i][-5:]}")
        print(f"Actions: {total_learning_data['actions'][i][-5:]}")

############################################################################
# Plots
############################################################################
plt.figure()
plt.plot(total_data['time'], np.array(total_data['actual_com_velocity'])[:,0], label='CoM_x_dot')
plt.plot(total_data['time'], np.array(total_data['desired_com_velocity'])[:,0], label='Target Vel')
plt.legend()
plt.grid(True) 
plt.show()