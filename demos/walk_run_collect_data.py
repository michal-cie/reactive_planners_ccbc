import pickle
import datetime
import numpy as np
from matplotlib import pyplot as plt
from walk_run_batch import walk_run_batch

if __name__ == '__main__':

    collect_learning_data = False
    visualize_pybullet = True
    conditioning_type = "contact" # Options: "contact", "des_vel"
    max_dv = 1.5#1.5
    min_vel = -1.0#-1.0
    max_vel = 1.3#1.3
    min_gait_time = 1#1
    max_gait_time = 3#3
    min_com_height = 0.30#0.29
    max_com_height = 0.30#0.31
    rand_init_cond = False
    force_disturbance = True

    total_learning_data = None
    total_data = None

    num_batch_goals = 5
    num_batches = 1
    num_batches_completed = 0
    num_fails = 0
    while num_batches_completed < num_batches:
        # try:
        batch_data, batch_learning_data = \
            walk_run_batch(max_dv, min_vel, max_vel, 
                    min_gait_time, max_gait_time, 
                    min_com_height, max_com_height,
                    num_batch_goals,
                    collect_learning_data = collect_learning_data,
                    learned_policy=None,
                    conditioning_type = conditioning_type,
                    rand_init_cond = rand_init_cond,
                    force_disturbance = force_disturbance,
                    visualize_pybullet = visualize_pybullet)
        # except Exception as e:
        #     print("Error in batch, skipping")
        #     continue
        if batch_data['fails'][0][-1] == True:
            print("Batch failed")
            num_fails += 1
            continue
        elif abs(batch_data['actual_com_velocity'][-1][0] - \
             batch_data['desired_com_velocity'][-1][0]) > 0.4:
            print("Batch failed, Vel diff too high")
            num_fails += 1
            continue
        num_batches_completed += 1
        if total_learning_data is None:
            total_learning_data = batch_learning_data
            total_data = batch_data
            continue

        for key in total_learning_data:
            total_learning_data[key].extend(batch_learning_data[key])
        for key in total_data:
            total_data[key].extend(batch_data[key])

# Save data
current_datetime = datetime.datetime.now()
datetime_str = current_datetime.strftime("%m_%d_%H_%M_%S")
file_path = f"/home/michal/projects/contact_con/devel/reactive_planners/demos/learning_data/data_X{str(int(num_batches_completed))}_{datetime_str}.pkl"

if collect_learning_data:
    with open(file_path, 'wb') as f:
        pickle.dump(total_learning_data, f)

############################################################################
# Plots
############################################################################
plt.figure()
plt.plot(total_data['time'], np.array(total_data['actual_com_velocity'])[:,0], label='CoM_x_dot')
plt.plot(total_data['time'], np.array(total_data['desired_com_velocity'])[:,0], label='Target Vel')
plt.legend()
plt.grid(True) 
plt.show()