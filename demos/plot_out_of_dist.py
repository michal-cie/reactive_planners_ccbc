import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

script_path = os.path.abspath(__file__)
script_dir_path = os.path.dirname(script_path)
eval_data_dir = f"{script_dir_path}/eval_data/out_of_dist"

# datasets = ['data_X25_07_08_19_25_48',
#             'data_X50_07_08_19_30_53', 
#             'data_X100_07_08_19_44_14', 
#             'data_X200_07_08_20_13_36',
#             'data_X400_07_10_14_42_59']#data_X400_07_10_14_42_59
datasets = ['data_X400_07_10_14_42_59']
epochs = [200]
angles = [60.0, 120.0, 240.0, 300.0]
train_type = ['']#['train_', '']
cond_type = ['pd_contact', 'pd_des_vel']

pd_contact_fails = []
pd_contact_time_to_fails = []
pd_contact_des_contact = []
pd_contact_act_contact = []
pd_contact_des_vel = []
pd_contact_act_vel = []

pd_des_vel_fails = []
pd_des_vel_time_to_fails = []
pd_des_vel_des_contact = []
pd_des_vel_act_contact = []
pd_des_vel_des_vel = []
pd_des_vel_act_vel = []


for dataset in datasets:
    for epoch in epochs:
        #for tt in train_type:
        for ct in cond_type:
            for angle in angles:
                policy = f"{ct}_states_ep{epoch}"
                eval_file_path = f"{eval_data_dir}/{str(angle)}_{policy}.pkl"
                with open(eval_file_path, 'rb') as f:
                    data_dict = pickle.load(f)
                if ct == 'pd_contact':
                    pd_contact_fails.append(data_dict['fails'])
                    pd_contact_time_to_fails.append(data_dict['time_to_fails'])
                    pd_contact_des_contact.append(data_dict['current_desired_contact'])
                    pd_contact_act_contact.append(data_dict['actual_contact'])
                    pd_contact_des_vel.append(data_dict['desired_com_velocity'])
                    pd_contact_act_vel.append(data_dict['actual_com_velocity'])
                elif ct == 'pd_des_vel':
                    pd_des_vel_fails.append(data_dict['fails'])
                    pd_des_vel_time_to_fails.append(data_dict['time_to_fails'])
                    pd_des_vel_des_contact.append(data_dict['current_desired_contact'])
                    pd_des_vel_act_contact.append(data_dict['actual_contact'])
                    pd_des_vel_des_vel.append(data_dict['desired_com_velocity'])
                    pd_des_vel_act_vel.append(data_dict['actual_com_velocity'])

# Fails over data regimes
pd_contact_fails = [sum(item for sublist in x for item in sublist) for x in pd_contact_fails]
pd_des_vel_fails = [sum(item for sublist in x for item in sublist) for x in pd_des_vel_fails]

des_contacts = [pd_contact_des_vel, pd_des_vel_des_vel]
act_contacts = [pd_contact_act_vel, pd_des_vel_act_vel]

contact_errors_x = [[],[]]
contact_stds_x = [[],[]]
box_errors_x = [[],[]]

contact_errors_y = [[],[]]
contact_stds_y = [[],[]]
box_errors_y = [[],[]]

for i in range(4):
    for j in range(2):
        des_x_contacts = np.array(des_contacts[j][i])[:,0]
        act_x_contacts = np.array(act_contacts[j][i])[:,0]
        des_y_contacts = np.array(des_contacts[j][i])[:,1]
        act_y_contacts = np.array(act_contacts[j][i])[:,1]

        # angles_des_radians = np.arctan(des_x_contacts[:, 1], des_x_contacts[:, 0])
        # angles_act_radians = np.arctan(act_x_contacts[:, 1], act_x_contacts[:, 0])

        # error = angles_act_radians - angles_des_radians
        error = abs(act_x_contacts - des_x_contacts)
        box_errors_x[j].append(error)
        contact_stds_x[j].append(np.std(abs(error)))
        contact_errors_x[j].append(abs(error).mean())

        error = abs(act_y_contacts - des_y_contacts)
        box_errors_y[j].append(error)
        contact_stds_y[j].append(np.std(abs(error)))
        contact_errors_y[j].append(abs(error).mean())

print(pd_contact_fails)
print(pd_des_vel_fails)

# Data for plotting
n_groups = 4
index = np.arange(n_groups)
bar_width = 0.20
opacity = 0.8

# Create the figure and axes
fig, ax = plt.subplots()

rects1 = ax.bar(index, contact_errors_x[0], bar_width, alpha=opacity, yerr=contact_stds_x[0],label='Contact X Vel', capsize=5)
rects2 = ax.bar(index + 1 * bar_width, contact_errors_x[1], bar_width, alpha=opacity, yerr=contact_stds_x[1], label='Desired Velocity X Vel', capsize=5)
rects3 = ax.bar(index + 2 * bar_width, contact_errors_y[0], bar_width, alpha=opacity, yerr=contact_stds_y[0],label='Contact Y Vel', capsize=5)
rects4 = ax.bar(index + 3 * bar_width, contact_errors_y[1], bar_width, alpha=opacity, yerr=contact_stds_y[1], label='Desired Velocity Y Vel', capsize=5)

# Add labels, title, and legend
ax.set_xlabel('Out of Distribution Angles (deg)')
ax.set_ylabel('Velocity (m/s)')
ax.set_title('Out of Distribution Velocity Error by Policy Type')
ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(['60','120','240','300'])
ax.legend()
plt.savefig(f"{eval_data_dir}/velocity_errors.png")

ticks = ['60','120','240','300']
fig, ax = plt.subplots()
box_pd_contact_err = box_errors_x[0]
box_pd_des_vel_err = box_errors_x[1]
box_pd_contact_err_y = box_errors_y[0]
box_pd_des_vel_err_y = box_errors_y[1]

positions1 = np.arange(len(box_pd_contact_err)) * 2.0 - 0.675
positions2 = np.arange(len(box_pd_des_vel_err)) * 2.0 - 0.225
positions3 = np.arange(len(box_pd_contact_err_y)) * 2.0 + 0.225
positions4 = np.arange(len(box_pd_des_vel_err_y)) * 2.0 + 0.675
width = 0.4
widths = [width]*4
bp1 = ax.boxplot(box_pd_contact_err,positions=positions1,widths=widths, patch_artist=True,
                 boxprops=dict(facecolor='blue'), medianprops=dict(color="black"),
                 whiskerprops=dict(color='blue'), capprops=dict(color='blue'), flierprops=dict(alpha=0.002, markersize=1,markerfacecolor='blue', marker='o'))
bp2 = ax.boxplot(box_pd_des_vel_err,positions=positions2,widths=widths, patch_artist=True,
                 boxprops=dict(facecolor='orange'), medianprops=dict(color="black"),
                 whiskerprops=dict(color='orange'), capprops=dict(color='orange'), flierprops=dict(alpha=0.002, markersize=1,markerfacecolor='orange', marker='o'))
bp3 = ax.boxplot(box_pd_contact_err_y,positions=positions3,widths=widths, patch_artist=True,
                 boxprops=dict(facecolor='green'), medianprops=dict(color="black"),
                 whiskerprops=dict(color='green'), capprops=dict(color='green'), flierprops=dict(alpha=0.002, markersize=1,markerfacecolor='green', marker='o'))
bp4 = ax.boxplot(box_pd_des_vel_err_y,positions=positions4,widths=widths, patch_artist=True,
                 boxprops=dict(facecolor='red'), medianprops=dict(color="black"),
                 whiskerprops=dict(color='red'), capprops=dict(color='red'), flierprops=dict(alpha=0.002, markersize=1,markerfacecolor='red', marker='o'))

ax.set_xlabel('Out of Distribution Angles (deg)')
ax.set_ylabel('Velocity (m/s)')
ax.set_title('Out of Distribution Velocity Error by Policy Type')
ax.set_xticks(np.arange(len(box_pd_contact_err)) * 2.0)
ax.set_xticklabels(ticks)
ax.legend([bp1["boxes"][0], bp2["boxes"][0],bp3["boxes"][0], bp4["boxes"][0]], ['Contact X Vel', 'Desired Velocity X Vel','Contact Y Vel', 'Desired Velocity Y Vel'], loc='upper left')

plt.savefig(f"{eval_data_dir}/velocity_errors_box.png")
plt.close()

# Data for plotting
n_groups = 4
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8

# Create the figure and axes
fig, ax = plt.subplots()

# Create bars for each category
rects1 = ax.bar(index, pd_contact_fails, bar_width, alpha=opacity, label='Contact')
rects2 = ax.bar(index + 1 * bar_width, pd_des_vel_fails, bar_width, alpha=opacity, label='Desired Velocity')

ax.set_xlabel('Out of Distribution Angles (deg)')
ax.set_ylabel('Number of Fails')
ax.set_title('Fails by Policy Type and Velocity Angles')
ax.set_xticks(index + 1 * bar_width)
ax.set_xticklabels(['60','120','240','300'])
ax.legend()
plt.savefig(f"{eval_data_dir}/fails.png")