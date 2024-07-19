import torch
import numpy as np
from matplotlib import pyplot as plt
from networks.networks2 import PolicyModel
from walk_run_sim_episode import run_simulation_episode

if __name__ == '__main__':

    policy = 'pd_des_vel_states_ep200'
    policy_dir = '/home/michal/projects/contact_con/devel/reactive_planners/demos/learned_policies'
    dataset_dir = 'data_X400_07_10_14_42_59'
    policy_file_path = f"{policy_dir}/{dataset_dir}/{policy}.pth"
    # policy_file_path = f"{policy_dir}/{policy}.pth"
    trained_policy = torch.load(policy_file_path)
    trained_policy = PolicyModel(trained_policy['model'])

    collect_learning_data = False
    visualize_pybullet = True
    conditioning_type = "pd_des_vel" # Options: "contact", "des_vel"
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

    #v_des = [[0.8, 0.0], [0.1, 0.0],[-0.5, 0.0],[0.5, 0.0]]
    v_des = [[0.4, 0.4], [0.4, 0.4], [0.4, 0.4]]
    #v_des = [[1.0, 0.0]]*num_batch_goals
    #v_des = [[0.0, 0.0],[0.2, 0.0],[-0.2, 0.0],[0.4, 0.0],[-0.4, 0.0],[0.8, 0.0],[-0.8, 0.0],[1.3, 0.0],[-1.3, 0.0]]
    v_des_times = [2]*num_batch_goals
    gait = ['walk', 'walk', 'walk', 'walk', 'walk', 'walk', 'walk', 'walk', 'walk']
    gait = ['run', 'run', 'run', 'run', 'run', 'run', 'run', 'run', 'run']
    gait = ['walk', 'run', 'walk', 'run', 'walk', 'run', 'walk', 'run', 'walk']
    gait = ['run', 'walk', 'run', 'walk']
    gait = ['run', 'walk', 'run']
    com_height = [0.30]*num_batch_goals

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

    trained_policy = None
    run_simulation_episode(gait, v_des, v_des_times, 
                           com_height, batch_data,
                           batch_learning_data,
                           collect_learning_data,
                           trained_policy,
                           conditioning_type,
                           rand_init_cond = rand_init_cond,
                           force_disturbance = force_disturbance,
                           visualize_pybullet = visualize_pybullet)