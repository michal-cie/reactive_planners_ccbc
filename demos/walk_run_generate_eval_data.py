import os
import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt
from networks.networks2 import PolicyModel
from walk_run_batch import walk_run_batch

if __name__ == '__main__':

    script_path = os.path.abspath(__file__)
    script_dir_path = os.path.dirname(script_path)
    policy_dir = f"{script_dir_path}/learned_policies"
    eval_data_dir = f"{script_dir_path}/eval_data/single_batch"

    datasets = ['data_X25_07_08_19_25_48',
                'data_X50_07_08_19_30_53', 
                'data_X100_07_08_19_44_14', 
                'data_X200_07_08_20_13_36',
                'data_X400_07_10_14_42_59']
    datasets = ['data_X400_07_10_14_42_59']
    epochs = [200]
    train_type = ['']
    cond_type = ['pd_two_contact', 'pd_two_contact_gait', 'pd_contact', 'pd_contact_gait', 'pd_des_vel']

    for dataset in datasets:
        for epoch in epochs:
            for tt in train_type:
                for ct in cond_type:
                    policy = f"{tt}{ct}_states_ep{epoch}"
                    policy_file_path = f"{policy_dir}/{dataset}/{policy}.pth"
                    eval_file_path = f"{eval_data_dir}/{dataset}/{policy}.pkl"

                    print(f"Generating evaluation data for {policy}")

                    trained_policy = torch.load(policy_file_path)
                    trained_policy = PolicyModel(trained_policy['model'])

                    collect_learning_data = False
                    visualize_pybullet = False
                    conditioning_type = ct
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

                    num_batch_goals = 1
                    num_batches = 100
                    num_batches_completed = 0
                    while num_batches_completed < num_batches:
                        try:
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
                        except Exception as e:
                            print("Error in batch, skipping")
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
                    with open(eval_file_path, 'wb') as file:
                        pickle.dump(total_data, file)