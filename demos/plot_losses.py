import os
import pickle
import matplotlib.pyplot as plt

script_path = os.path.abspath(__file__)
script_dir_path = os.path.dirname(script_path)

cond_type = ['pd_contact_gait', 'pd_contact', 'pd_des_vel']

data_files = ['data_X69_06_24_12_17_50',
              'data_X132_06_24_12_09_11', 
              'data_X237_06_24_12_18_11', 
              'data_X513_06_24_12_34_14',
              'data_X939_06_24_15_07_42']
for i in range(5):
    losses_path = f'learned_policies/{data_files[i]}/losses'


    # for cd in cond_type:
    #     total_path = f"{script_dir_path}/{losses_path}/{cd}_{data_files[0]}.pkl"

    #     with open(total_path, 'rb') as f:
    #         losses_dict = pickle.load(f)
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(losses_dict['train_losses'][:100], label='Training Loss')
    #     plt.plot(losses_dict['val_losses'][:100], label='Validation Loss')
    #     plt.title('Contact Conditioned Training and Validation Losses')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.yscale('log')
    #     plt.legend()
    # plt.show()

    ccg_total_path = f"{script_dir_path}/{losses_path}/{cond_type[0]}_{data_files[i]}.pkl"
    cc_total_path = f"{script_dir_path}/{losses_path}/{cond_type[1]}_{data_files[i]}.pkl"
    vc_total_path = f"{script_dir_path}/{losses_path}/{cond_type[2]}_{data_files[i]}.pkl"

    with open(ccg_total_path, 'rb') as f:
        ccg_losses_dict = pickle.load(f)
    with open(cc_total_path, 'rb') as f:
        cc_losses_dict = pickle.load(f)
    with open(vc_total_path, 'rb') as f:
        vc_losses_dict = pickle.load(f)
    plt.figure(figsize=(10, 6))
    plt.plot(ccg_losses_dict['train_losses'][:100], label='Training Loss CCG')
    plt.plot(ccg_losses_dict['val_losses'][:100], label='Validation Loss CCG')
    plt.plot(cc_losses_dict['train_losses'][:100], label='Training Loss CC')
    plt.plot(cc_losses_dict['val_losses'][:100], label='Validation Loss CC')
    plt.plot(vc_losses_dict['train_losses'][:100], label='Training Loss VC')
    plt.plot(vc_losses_dict['val_losses'][:100], label='Validation Loss VC')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
plt.show()