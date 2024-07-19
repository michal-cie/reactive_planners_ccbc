import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

script_path = os.path.abspath(__file__)
script_dir_path = os.path.dirname(script_path)
eval_data_dir = f"{script_dir_path}/eval_data/single_batch"

datasets = ['data_X25_07_08_19_25_48',
            'data_X50_07_08_19_30_53', 
            'data_X100_07_08_19_44_14', 
            'data_X200_07_08_20_13_36',
            'data_X400_07_10_14_42_59']#data_X400_07_10_14_42_59
# datasets = ['data_X200_07_08_20_13_36']
epochs = [200]
train_type = ['']#['train_', '']
#cond_type = ['pd_two_contact' 'pd_contact','pd_des_vel']
cond_type = ['pd_two_contact', 'pd_contact', 'pd_des_vel']
# policy = f"{train_type[1]}{cond_type[1]}_states_ep{epochs[0]}"
# eval_file_path = f"{eval_data_dir}/{datasets[1]}/{policy}.pkl"


# with open(eval_file_path, 'rb') as f:
#     data_dict = pickle.load(f)

# # print(data_dict['desired_com_velocity'])
# # print(data_dict['actual_com_velocity'])

# # print(data_dict['current_desired_contact'])
# # print(data_dict['actual_contact'])

# print(data_dict['fails'])
# print(data_dict['time_to_fails'])
# print(data_dict['total_distance'])

pd_contact_fails = []
pd_contact_time_to_fails = []
pd_contact_des_contact = []
pd_contact_act_contact = []
pd_contact_des_vel = []
pd_contact_act_vel = []

pd_contact_gait_fails = []
pd_contact_gait_time_to_fails = []
pd_contact_gait_des_contact = []
pd_contact_gait_act_contact = []
pd_contact_gait_des_vel = []
pd_contact_gait_act_vel = []

pd_two_contact_fails = []
pd_two_contact_time_to_fails = []
pd_two_contact_des_contact = []
pd_two_contact_act_contact = []
pd_two_contact_des_vel = []
pd_two_contact_act_vel = []

pd_two_contact_gait_fails = []
pd_two_contact_gait_time_to_fails = []
pd_two_contact_gait_des_contact = []
pd_two_contact_gait_act_contact = []
pd_two_contact_gait_des_vel = []
pd_two_contact_gait_act_vel = []

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
            policy = f"{ct}_states_ep{epoch}"
            eval_file_path = f"{eval_data_dir}/{dataset}/{policy}.pkl"
            with open(eval_file_path, 'rb') as f:
                data_dict = pickle.load(f)
            if ct == 'pd_contact':
                pd_contact_fails.append(data_dict['fails'])
                pd_contact_time_to_fails.append(data_dict['time_to_fails'])
                pd_contact_des_contact.append(data_dict['current_desired_contact'])
                pd_contact_act_contact.append(data_dict['actual_contact'])
                pd_contact_des_vel.append(data_dict['desired_com_velocity'])
                pd_contact_act_vel.append(data_dict['actual_com_velocity'])
            # elif ct == 'pd_contact_gait':
            #     pd_contact_gait_fails.append(data_dict['fails'])
            #     pd_contact_gait_time_to_fails.append(data_dict['time_to_fails'])
            #     pd_contact_gait_des_contact.append(data_dict['current_desired_contact'])
            #     pd_contact_gait_act_contact.append(data_dict['actual_contact'])
            #     pd_contact_gait_des_vel.append(data_dict['desired_com_velocity'])
            #     pd_contact_gait_act_vel.append(data_dict['actual_com_velocity'])
            elif ct == 'pd_two_contact':
                pd_two_contact_fails.append(data_dict['fails'])
                pd_two_contact_time_to_fails.append(data_dict['time_to_fails'])
                pd_two_contact_des_contact.append(data_dict['current_desired_contact'])
                pd_two_contact_act_contact.append(data_dict['actual_contact'])
                pd_two_contact_des_vel.append(data_dict['desired_com_velocity'])
                pd_two_contact_act_vel.append(data_dict['actual_com_velocity'])
            # elif ct == 'pd_two_contact_gait':
            #     pd_two_contact_gait_fails.append(data_dict['fails'])
            #     pd_two_contact_gait_time_to_fails.append(data_dict['time_to_fails'])
            #     pd_two_contact_gait_des_contact.append(data_dict['current_desired_contact'])
            #     pd_two_contact_gait_act_contact.append(data_dict['actual_contact'])
            #     pd_two_contact_gait_des_vel.append(data_dict['desired_com_velocity'])
            #     pd_two_contact_gait_act_vel.append(data_dict['actual_com_velocity'])
            elif ct == 'pd_des_vel':
                pd_des_vel_fails.append(data_dict['fails'])
                pd_des_vel_time_to_fails.append(data_dict['time_to_fails'])
                pd_des_vel_des_contact.append(data_dict['current_desired_contact'])
                pd_des_vel_act_contact.append(data_dict['actual_contact'])
                pd_des_vel_des_vel.append(data_dict['desired_com_velocity'])
                pd_des_vel_act_vel.append(data_dict['actual_com_velocity'])

# Fails over data regimes
box_data_pd_contact_fails = [[item for sublist in x for item in sublist]for x in pd_contact_time_to_fails]
pd_contact_fails = [sum(item for sublist in x for item in sublist) for x in pd_contact_fails]

# pd_contact_gait_fails = [sum(item for sublist in x for item in sublist) for x in pd_contact_gait_fails]
box_data_pd_two_contact_fails = [[item for sublist in x for item in sublist ]for x in pd_two_contact_time_to_fails]
pd_two_contact_fails = [sum(item for sublist in x for item in sublist) for x in pd_two_contact_fails]
# pd_two_contact_gait_fails = [sum(item for sublist in x for item in sublist) for x in pd_two_contact_gait_fails]
box_data_pd_des_vel_fails = [[item for sublist in x for item in sublist ]for x in pd_des_vel_time_to_fails]
pd_des_vel_fails = [sum(item for sublist in x for item in sublist) for x in pd_des_vel_fails]

# des_contacts = [pd_contact_des_contact, pd_contact_gait_des_contact, pd_two_contact_des_contact, pd_two_contact_gait_des_contact, pd_des_vel_des_contact]
# act_contacts = [pd_contact_act_contact, pd_contact_gait_act_contact, pd_two_contact_act_contact, pd_two_contact_gait_act_contact, pd_des_vel_act_contact]

# pd_contact_contact_error = []
# pd_contact_gait_contact_error = []
# pd_two_contact_contact_error = []
# pd_two_contact_gait_contact_error = []
# pd_des_vel_contact_error = []
# contact_errors = [pd_contact_contact_error, pd_contact_gait_contact_error , pd_two_contact_contact_error, pd_two_contact_gait_contact_error, pd_des_vel_contact_error]

# for i in range(5):
#     for j in range(5):
#         des_x_contacts = np.array(des_contacts[j][i])[:,0]
#         act_x_contacts = np.array(act_contacts[j][i])[:,0]
#         error = act_x_contacts - des_x_contacts
#         contact_errors[j].append((error**2).mean())

pd_contact_fails = [sum(item for sublist in x for item in sublist)/len(x) for x in pd_contact_time_to_fails]
# pd_contact_gait_fails = [sum(item for sublist in x for item in sublist)/len(x) for x in pd_contact_gait_time_to_fails]
pd_two_contact_fails = [sum(item for sublist in x for item in sublist)/len(x) for x in pd_two_contact_time_to_fails]
# pd_two_contact_gait_fails = [sum(item for sublist in x for item in sublist)/len(x) for x in pd_two_contact_gait_time_to_fails]
pd_des_vel_fails = [sum(item for sublist in x for item in sublist)/len(x) for x in pd_des_vel_time_to_fails]

print(pd_contact_fails)
# print(pd_contact_gait_fails)
print(pd_two_contact_fails)
# print(pd_two_contact_gait_fails)
print(pd_des_vel_fails)

# Data for plotting
n_groups = 5
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8

# Create the figure and axes
fig, ax = plt.subplots()

# Create bars for each category
rects1 = ax.bar(index, pd_contact_fails, bar_width, yerr=np.std(pd_contact_fails), alpha=opacity, label='Contact', capsize=5)
# rects2 = ax.bar(index + bar_width, pd_contact_gait_fails, bar_width, alpha=opacity, label='Contact Gait')
rects3 = ax.bar(index + 1 * bar_width, pd_two_contact_fails, bar_width, yerr=np.std(pd_two_contact_fails), alpha=opacity, label='Two Contact', capsize=5)
# rects4 = ax.bar(index + 3 * bar_width, pd_two_contact_gait_fails, bar_width, alpha=opacity, label='Two Contact Gait')
rects5 = ax.bar(index + 2 * bar_width, pd_des_vel_fails, bar_width, yerr=np.std(pd_des_vel_fails), alpha=opacity, label='Desired Velocity', capsize=5)

# rects1 = ax.bar(index, contact_errors[0], bar_width, alpha=opacity, label='Contact')
# rects2 = ax.bar(index + bar_width, contact_errors[1], bar_width, alpha=opacity, label='Contact Gait')
# rects3 = ax.bar(index + 2 * bar_width, contact_errors[2], bar_width, alpha=opacity, label='Two Contact')
# rects4 = ax.bar(index + 3 * bar_width, contact_errors[3], bar_width, alpha=opacity, label='Two Contact Gait')
# rects5 = ax.bar(index + 4 * bar_width, contact_errors[4], bar_width, alpha=opacity, label='Desired Velocity')

# Add labels, title, and legend
# ax.set_xlabel('Rollouts in Training Data')
# ax.set_ylabel('Distance (m)')
# ax.set_title('Contact Error by Policy Type and Rollouts in Training Data')
# ax.set_xticks(index + 2 * bar_width)
# ax.set_xticklabels(['25','50','100','200'])
# ax.legend()
ax.set_xlabel('Rollouts in Training Data')
ax.set_ylabel('Time (s)')
ax.set_title('Time to Failure by Policy Type and Rollouts in Training Data')
ax.set_xticks(index + 1 * bar_width)
ax.set_xticklabels(['25','50','100','200','400'])

target_y_value = 2  # Set this to your target value
ax.axhline(y=target_y_value, color='red', linewidth=1, label='Expected Duration')
ax.legend()

# # Label the heights of the bars
# def label_bars(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')

# label_bars(rects1)
# label_bars(rects2)
# label_bars(rects3)
# label_bars(rects4)
# label_bars(rects5)

plt.savefig(f"{eval_data_dir}/time_to_fails.png")

# plt.figure()
# plt.boxplot([contact_dists_expert_diff, des_vel_dists_expert_diff])
# plt.xticks([1, 2], ['Contact', 'Des_vel'])
# plt.ylabel('Distance Difference')
# plt.title('Total Distance Differences with Expert')

ticks = ['25', '50', '100', '200', '400']
fig, ax = plt.subplots()
# Creating box plot
# position controls where each group of boxplots appears

# box_data_pd_contact_fails = [[item for sublist in x for item in sublist ]for x in pd_contact_fails]
# box_data_pd_two_contact_fails = [[item for sublist in x for item in sublist ]for x in pd_two_contact_fails]
# box_data_pd_des_vel_fails = [[item for sublist in x for item in sublist ]for x in pd_des_vel_fails]
print(len(box_data_pd_contact_fails[0]))
print(len(box_data_pd_contact_fails[1]))
print(len(box_data_pd_contact_fails[2]))
print(len(box_data_pd_contact_fails[3]))
print(len(box_data_pd_contact_fails[4]))
positions1 = np.arange(len(box_data_pd_contact_fails)) * 2.0 - 0.6
positions2 = np.arange(len(box_data_pd_two_contact_fails)) * 2.0
positions3 = np.arange(len(box_data_pd_des_vel_fails)) * 2.0 + 0.6
print(len(box_data_pd_contact_fails))
bp1 = ax.boxplot(box_data_pd_contact_fails,positions=positions1, patch_artist=True,
                 boxprops=dict(facecolor='blue'), medianprops=dict(color="black"),
                 whiskerprops=dict(color='blue'), capprops=dict(color='blue'), flierprops=dict(alpha=0.002, markersize=1,markerfacecolor='blue', marker='o'))
bp2 = ax.boxplot(box_data_pd_two_contact_fails,positions=positions2, patch_artist=True,
                 boxprops=dict(facecolor='orange'), medianprops=dict(color="black"),
                 whiskerprops=dict(color='orange'), capprops=dict(color='orange'), flierprops=dict(alpha=0.002, markersize=1,markerfacecolor='orange', marker='o'))
bp3 = ax.boxplot(box_data_pd_des_vel_fails,positions=positions3, patch_artist=True,
                 boxprops=dict(facecolor='green'), medianprops=dict(color="black"),
                 whiskerprops=dict(color='green'), capprops=dict(color='green'), flierprops=dict(alpha=0.002, markersize=1,markerfacecolor='green', marker='o'))

ax.set_xlabel('Rollouts in Training Data')
ax.set_ylabel('Time (s)')
ax.set_title('Time to Failure by Policy Type and Rollouts in Training Data')
ax.set_xticks(np.arange(len(box_data_pd_two_contact_fails)) * 2.0)
ax.set_xticklabels(ticks)
target_y_value = 2
line = ax.axhline(y=target_y_value, color='red', linewidth=1, label='Expected Duration')
line_legend = Line2D([0], [0], color='red', linewidth=1, linestyle='-')
ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], line_legend], ['Contact', 'Two Contact', 'Desired Velocity', 'Expected Duration'], loc='upper left')

plt.savefig(f"{eval_data_dir}/time_to_fails_box.png")
plt.close()