import random
import numpy as np
from walk_run_sim_episode import run_simulation_episode

def walk_run_batch(max_dv, min_vel, max_vel, 
                   min_gait_time, max_gait_time, 
                   min_com_height, max_com_height,
                   num_goals,
                   collect_learning_data = True,
                   learned_policy = None,
                   conditioning_type = "contact",
                   rand_init_cond = False,
                   force_disturbance = False,
                   visualize_pybullet = True):
    
    batch_learning_data = {'contact_states': [],
                           'two_contact_states': [],
                           'contact_gait_states': [],
                           'two_contact_gait_states': [],
                           'des_vel_states': [],
                           'actions': [],
                           'pd_actions': []}

    batch_data = {
        'time': [],
        'desired_com_velocity': [],
        'desired_gait': [],
        'actual_com_velocity': [],
        'next_desired_contact': [],
        'current_desired_contact': [],
        'actual_contact': [],
        'actual_com_position': [],
        'torques': [],
        'fails': [],
        'time_to_fails': [],
        'total_distance': [],
    }

    v_des = []
    v_des_times = []
    gait = []
    com_height = []
    previous_v_des = 0.0#random.uniform(min_vel, max_vel)  # Initialize with a random value within the bounds

    for _ in range(num_goals):
        if random.choice([True, False]):
            # Compute the next v_des ensuring it is within the 0.5 range of the previous value
            # lower_bound = max(previous_v_des - max_dv, min_vel)
            # upper_bound = min(previous_v_des + max_dv, max_vel)
            # v_des_ = random.uniform(lower_bound, upper_bound)
            v = random.uniform(0.7, 0.7)
            v_des_ = [v * np.cos(max_vel), 0 * np.sin(max_vel)]
            time = random.uniform(min_gait_time, max_gait_time)
            com_height_ = random.uniform(min_com_height, max_com_height)
            v_des.append(v_des_)
            v_des_times.append(2)
            gait.append('walk')
            com_height.append(com_height_)
        else:
            # Compute the next v_des ensuring it is within the 0.5 range of the previous value
            # lower_bound = max(previous_v_des - max_dv, min_vel)
            # upper_bound = min(previous_v_des + max_dv, max_vel)
            # v_des_ = random.uniform(lower_bound, upper_bound)
            v = random.uniform(0.7, 0.7)
            v_des_ = [v * np.cos(max_vel), 0 * np.sin(max_vel)]
            time = random.uniform(min_gait_time, max_gait_time)
            com_height_ = random.uniform(min_com_height, max_com_height)
            v_des.append(v_des_)
            v_des_times.append(2)
            gait.append('run')
            com_height.append(com_height_)

        # previous_v_des = v_des_

    run_simulation_episode(gait, v_des, v_des_times, 
                           com_height, batch_data,
                           batch_learning_data,
                           collect_learning_data,
                           learned_policy,
                           conditioning_type,
                           rand_init_cond = rand_init_cond,
                           force_disturbance = force_disturbance,
                           visualize_pybullet = visualize_pybullet)

    return batch_data, batch_learning_data
