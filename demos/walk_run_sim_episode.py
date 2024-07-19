import torch
import random
import numpy as np
import pybullet as p
import pinocchio as se3 # type: ignore
from robot_properties_bolt.config import BoltConfig
from robot_properties_bolt.bolt_wrapper import BoltRobot
from mim_control.robot_centroidal_controller import RobotCentroidalController
from mim_control.robot_impedance_controller import RobotImpedanceController
from reactive_planners_cpp import DcmReactiveStepper # type: ignore
from scipy.spatial.transform import Rotation as R
from bullet_utils.env import BulletEnvWithGround


def zero_cnt_gain(kp, cnt_array):
    gain = np.array(kp).copy()
    for i, v in enumerate(cnt_array):
        if v == 1:
            gain[3 * i : 3 * (i + 1)] = 0.0
    return gain

def yaw(q):
    #print(q)
    return np.array(
        R.from_quat([np.array(q)[3:7]]).as_euler("xyz", degrees=False)
    )[0, 2]


def external_force(com):
    force = np.array(
        [
            (random() - 0.5) * 7000,
            (random() - 0.5) * 7000,
            (random() - 0.5) * 2500,
            ]
    )
    p.applyExternalForce(
        objectUniqueId=robot.robotId,
        linkIndex=-1,
        forceObj=force,
        posObj=[com[0], com[1], com[2]],
        flags=p.WORLD_FRAME,
    )

def run_simulation_episode(gaits, v_dess, v_des_times, 
                           com_heights, batch_data,
                           batch_learning_data,
                           collect_learning_data = True,
                           learned_policy = None,
                           conditioning_type = "contact",
                           rand_init_cond = False,
                           force_disturbance = False,
                           visualize_pybullet = True):

    if visualize_pybullet:
        server = p.GUI
    else:
        server = p.DIRECT
    env = BulletEnvWithGround(server)
    robot = env.add_robot(BoltRobot())

    tau = np.zeros(6)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
    p.resetDebugVisualizerCamera(1., 50, -50, (0.0, 0.0, 0.0))
    # p.resetDebugVisualizerCamera(1., 120, -35, (0.0, 0.0, 0.0))
    p.setTimeStep(0.001)
    p.setRealTimeSimulation(0)
    p.removeAllUserParameters()
    p.removeAllUserDebugItems()

    # Set the sphere's properties
    radius = 0.015  # Radius of the sphere
    start_position = [0, 0, 0]  # Starting position of the sphere (x, y, z)
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])  # Starting orientation of the sphere

    # Create the visual shape for the sphere
    visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1])  # Red color

    # Create the multi-body for the sphere without a collision shape
    sphere_id = p.createMultiBody(
        baseMass=0,  # Mass is set to 0 for non-collision visual objects
        baseCollisionShapeIndex=-1,  # No collision shape
        baseVisualShapeIndex=visual_shape_id,
        basePosition=start_position,
        baseOrientation=start_orientation
    )

    for ji in range(8):
        p.changeDynamics(
            robot.robotId,
            ji,
            linearDamping=0.04,
            angularDamping=0.04,
            restitution=0.0,
            lateralFriction=4.0,
            spinningFriction=5.6,
        )
    xy_rand = np.random.uniform(-1.0, 1.0, 2)
    z_rand = np.random.uniform(0.0, 0.02)
    rpy_rand = np.random.uniform(-(5 * np.pi)/180, (5 * np.pi)/180, size=3)
    rotation_matrix = se3.rpy.rpyToMatrix(rpy_rand[0], rpy_rand[1], rpy_rand[2])
    quaternion = se3.Quaternion(rotation_matrix)
    q_rand = np.random.uniform(-(5 * np.pi)/180, (5 * np.pi)/180, 6)
    
    q_vel_rand = np.random.uniform(-0.1, 0.1, 12)
    q = np.matrix(BoltConfig.initial_configuration).T
    if rand_init_cond:
        q[:3] += np.array([xy_rand[0], xy_rand[1], z_rand]).reshape(-1, 1)
        q[3:7] = quaternion.coeffs().reshape(-1, 1)
        q[7:] += q_rand.reshape(-1, 1)

    qdot = np.matrix(BoltConfig.initial_velocity).T
    if rand_init_cond:
        qdot += np.matrix(q_vel_rand.astype(np.int64)).reshape(-1, 1)

    robot.reset_state(q, qdot)
    total_mass = sum([i.mass for i in robot.pin_robot.model.inertias[1:]])
    warmup = 10
    kp = np.array([150.0, 150.0, 150.0, 150.0, 150.0, 150.0])
    kd = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    
    robot_config = BoltConfig()
    config_file = robot_config.ctrl_path
    bolt_leg_ctrl = RobotImpedanceController(robot, config_file)
    centr_controller = RobotCentroidalController(
        robot_config,
        mu=1,
        kc=[0, 0, 200],
        dc=[0, 0, 20],
        kb=[30, 5, 0],
        db=[40, 20, 0],
        qp_penalty_lin=[1, 1, 1e6],
        qp_penalty_ang=[1e6, 1e6, 1],
    )

    # Episode Parameters
    v_des = [v_dess[0][0], v_dess[0][1], 0.0]
    v_des_time = v_des_times[0]
    v_des_count = 0
    total_sim_time = sum(v_des_times)
    gait = gaits[0]
    prev_gait = gait
    #text_id = p.addUserDebugText(f"Gait: {gait}", [0, 0, 1], textColorRGB=[1, 0, 0], textSize=1.5)
    com_height = com_heights[0]

    # Walk/Run Parameters
    if gait == 'walk':
        omega = 5.7183913822
        t_s = 0.25
    elif gait == 'run':
        omega = 10.18
        t_s = 0.1

    l_min = -0.2
    l_max = 0.2
    w_min = -0.2
    w_max = 0.2
    t_min = 0.1
    t_max = 0.2
    l_p = 0.1235

    if random.choice([True, False]):
        is_left_leg_in_contact = True
    else:
        is_left_leg_in_contact = False

    weight = [1, 1, 5, 1000, 1000, 5, 10000000, 10000000, 10000000, 10000000]
    mid_air_foot_height = 0.05
    control_period = 0.001
    planner_loop = 0.001
    # Left foot and right foot positions
    x_des_local = [
        q[0].item(),
        q[1].item() + 0.02,
        0.0,
        q[0].item(),
        q[1].item() - 0.02,
        0.0,
        ]
    dcm_reactive_stepper = DcmReactiveStepper()
    dcm_reactive_stepper.initialize(
        is_left_leg_in_contact,
        l_min,
        l_max,
        w_min,
        w_max,
        t_min,
        t_max,
        l_p,
        com_height,
        t_s,
        weight,
        mid_air_foot_height,
        control_period,
        planner_loop,
        x_des_local[:3],
        x_des_local[3:],
        v_des)

    dcm_reactive_stepper.set_new_motion(com_height, omega, t_s)
    dcm_reactive_stepper.set_desired_com_velocity(v_des)

    x_com = np.zeros((3, 1))
    x_com[:] = [[0.0], [0.0], [com_height]]
    xd_com = np.zeros((3, 1))

    cnt_array = [1, 1]
    time = 0
    control_time = 0
    open_loop = True
    duration_of_stance_phase = 0.0

    dcm_force = [0.0, 0.0, 0.0]
    offset = 0.025

    dcm_reactive_stepper.start()
    record_plot = False

    if collect_learning_data:
        start_data_collection = np.random.uniform(0.0, total_sim_time)
        start_data_collection = start_data_collection
    else:
        start_data_collection = 0.0

    start_phase = None
    start_step = False

    if force_disturbance:
        apply_force_disturbance = random.choice([True, False])
        t_force_start = np.random.uniform(0.0, total_sim_time - 0.5)
        t_force_dur = np.random.uniform(0.0, 0.25)
        t_force_end = t_force_start + t_force_dur
    else:
        apply_force_disturbance = False

    force_direction = np.random.normal(0, 1, 3)
    force_direction /= np.linalg.norm(force_direction)

    force_mag = np.random.uniform(0, 10)
    force = force_direction * force_mag

    # Extra variables for data collection
    prev_des_contact = [0.0, 0.0, 0.0]
    des_contact = [0.0, 0.0, 0.0]
    total_distance = [0.0, 0.0, 0.0]
    prev_base_pos = [0.0, 0.0, 0.0]
    failed = False

    for i in range(int(total_sim_time/0.001)):
        if control_time > v_des_time and v_des_count < len(v_dess) - 1:
            v_des_count += 1
            v_des = [v_dess[v_des_count][0], v_dess[v_des_count][1], 0.0]
            gait = gaits[v_des_count]
            com_height = com_heights[v_des_count]
            v_des_time += v_des_times[v_des_count]
            dcm_reactive_stepper.set_desired_com_velocity(v_des)
            if gait == 'run':
                omega = 10.18
                t_s = 0.1
                dcm_reactive_stepper.set_new_motion(com_height, omega, t_s)
            elif gait == 'walk':
                omega = 5.7183913822
                t_s = 0.25
                dcm_reactive_stepper.set_new_motion(com_height, omega, t_s)

        last_qdot = qdot.copy()
        q, qdot = robot.get_state()

        q_fk = q.copy()
        se3.framesForwardKinematics(robot.pin_robot.model, robot.pin_robot.data, q_fk)
        se3.framesForwardKinematics(bolt_leg_ctrl.robot.pin_robot.model, bolt_leg_ctrl.robot.pin_robot.data, q_fk)

        if i%10 == 0:
            if i != 0:
                p.removeUserDebugItem(arrow_id)
            p.resetBasePositionAndOrientation(sphere_id, dcm_reactive_stepper.get_next_support_foot_position().tolist(), [0,0,0,1])
            p.resetDebugVisualizerCamera(1., 50, -50, (q[0].item(), q[1].item(), 0.0))
            start = [q[0].item(), q[1].item(), q[2].item()]
            end = [q[0].item() + v_des[0], q[1].item() + v_des[1], q[2].item() + v_des[2]]
            arrow_color = [1, 0, 0]  # Red color
            line_width = 5
            arrow_id = p.addUserDebugLine(start, end, arrow_color, lineWidth=line_width)
            # p.resetDebugVisualizerCamera(1., 120, -35, (q[0].item(), q[1].item(), 0.0))

        base_pos = [q[0].item(), q[1].item(), q[2].item()]
        total_distance[0] += abs(base_pos[0] - prev_base_pos[0])
        total_distance[1] += abs(base_pos[1] - prev_base_pos[1])
        total_distance[2] += abs(base_pos[2] - prev_base_pos[2])

        x_com = robot.pin_robot.com(q, qdot)[0]
        xd_com = robot.pin_robot.com(q, qdot)[1]

        if warmup <= i:
            # Unpack impedance controller for each ee
            left = bolt_leg_ctrl.imp_ctrl_array[0]
            right = bolt_leg_ctrl.imp_ctrl_array[1]
            left_foot_location = np.array(
                left.pin_robot.data.oMf[left.frame_end_idx].translation
            ).reshape(-1)
            right_foot_location = np.array(
                right.pin_robot.data.oMf[right.frame_end_idx].translation
            ).reshape(-1)
            left_foot_vel = np.array(
                se3.SE3(
                    left.pin_robot.data.oMf[left.frame_end_idx]
                )
                * se3.computeFrameJacobian(
                    robot.pin_robot.model,
                    robot.pin_robot.data,
                    q,
                    left.frame_end_idx,
                ).dot(qdot)[0:3]
            )
            right_foot_vel = np.array(
                se3.SE3(
                    right.pin_robot.data.oMf[right.frame_end_idx]
                )
                * se3.computeFrameJacobian(
                    robot.pin_robot.model,
                    robot.pin_robot.data,
                    q,
                    right.frame_end_idx,
                ).dot(qdot)[0:3]
            )

            is_left_leg_in_contact = dcm_reactive_stepper.get_is_left_leg_in_contact()

            dcm_reactive_stepper.run(
                time,
                [
                    left_foot_location[0],
                    left_foot_location[1],
                    left_foot_location[2] - offset,
                    ],
                [
                    right_foot_location[0],
                    right_foot_location[1],
                    right_foot_location[2] - offset,
                    ],
                left_foot_vel,
                right_foot_vel,
                x_com,
                xd_com,
                yaw(q),
                not open_loop,
            )

            is_left_leg_in_contact = dcm_reactive_stepper.get_is_left_leg_in_contact()

            x_des_local = []
            x_des_local.extend(
                dcm_reactive_stepper.get_left_foot_position().copy()
            )
            x_des_local.extend(
                dcm_reactive_stepper.get_right_foot_position().copy()
            )
            cnt_array = dcm_reactive_stepper.get_contact_phase()
            current_kessay = x_com[2] + (xd_com[2] / omega)
            u_t = -dcm_reactive_stepper.get_dcm_offset() + x_com + (xd_com / omega)
            if cnt_array[0] == cnt_array[1]: #== False
                if dcm_reactive_stepper.get_is_left_leg_in_contact():
                    x_des_local[:3] = x_com
                    x_des_local[2] = (x_com[2] - 0.2) * (-current_kessay + take_off_kessay) / (take_off_kessay - 0.2)
                    x_des_local[3:] = [u_t[0],
                                       u_t[1],
                                       (0.1) * (current_kessay - 0.2) / (take_off_kessay - 0.2)]

                    kp = np.array([10.0, 10.0, 75.0, 250.0, 150.0, 150.0])
                    kd = [.5, .5, 5., 5.0, 5.0, 5.0]
                else:
                    x_des_local[3:] = x_com
                    x_des_local[5] = (x_com[2] - 0.2) * (-current_kessay + take_off_kessay) / (take_off_kessay - 0.2)
                    x_des_local[:3] = [u_t[0],
                                       u_t[1],
                                       (0.1) * (current_kessay - 0.2) / (take_off_kessay - 0.2)]

                    kp = np.array([250.0, 150.0, 150.0, 10.0, 10.0, 75.0])
                    kd = [5.0, 5.0, 5.0, .5, .5, 5.]
            elif cnt_array[0] == 1:
                take_off_kessay = x_com[2] + (xd_com[2] / omega)

                x_des_local[3:] = [u_t[0],
                                   u_t[1],
                                   0.1]
                x_des_local[:3] = dcm_reactive_stepper.get_current_support_foot_position().copy()
                if omega == 5.7183913822:
                    x_des_local[5] = 0.1 / 0.2 * (0.2 - dcm_reactive_stepper.get_time_from_last_step_touchdown())

                if omega == 5.7183913822:
                    kp = 4 * np.array([0.0, 0.0, 0.0, 30.0, 30.0, 75.0])
                    kd = 4 * [.0, .0, .0, 4., 4., 10.]
                else:
                    kp = 1 * np.array([0.0, 0.0, 0.0, 10.0, 10.0, 75.0])
                    kd = 1 * [.0, .0, .0, 1., 1., 10.]

            else:
                take_off_kessay = x_com[2] + (xd_com[2] / omega)

                x_des_local[:3] = [u_t[0],
                                   u_t[1],
                                   0.1]

                x_des_local[3:] = dcm_reactive_stepper.get_current_support_foot_position().copy()
                if omega == 5.7183913822:
                    x_des_local[2] = 0.1 / 0.2 * (0.2 - dcm_reactive_stepper.get_time_from_last_step_touchdown())

                if omega == 5.7183913822:
                    kp = 4 * np.array([30.0, 30.0, 75.0, 0.0, 0.0, 0.0])
                    kd = 4 * [4., 4., 10., .0, .0, .0]
                else:
                    kp = 1 * np.array([10.0, 10.0, 75.0, 0.0, 0.0, 0.0])
                    kd = 1 * [1., 1., 10., .0, .0, .0]


            if open_loop:
                x_des_local[2] += offset
                x_des_local[5] += offset

                if dcm_reactive_stepper.get_time_from_last_step_touchdown() == 0:
                    pass
            time += 0.001

        for j in range(2):
            imp = bolt_leg_ctrl.imp_ctrl_array[j]
            x_des_local[3 * j : 3 * (j + 1)] -= imp.pin_robot.data.oMf[
                imp.frame_root_idx
            ].translation

            if record_plot:
                pass
        if warmup <= i:
            com = dcm_reactive_stepper.get_com()
            v_com = dcm_reactive_stepper.get_v_com()
            a_com = dcm_reactive_stepper.get_a_com()
        else:
            com = [0.0, 0.0, com_height]
            v_com = [0.0, 0.0, 0.0]
            a_com = [0.0, 0.0, 0.0]
        if record_plot:
            pass
        w_com = centr_controller.compute_com_wrench(q.copy(), qdot.copy(), com, v_com,
                                                    [0, 0.0, 0, 1.0], [0.0, 0.0, 0.0],)
        w_com[0] += a_com[0] * total_mass
        w_com[1] += a_com[1] * total_mass
        w_com[2] += a_com[2] * total_mass#Lhum TODO add it to all the directions

        F = centr_controller.compute_force_qp(q, qdot, cnt_array, w_com)

        if cnt_array[0] == 0 and cnt_array[1] == 0:
            des_vel = [0., 0., 0., 0., 0., 0.]
        else:
            des_vel = np.concatenate((dcm_reactive_stepper.get_left_foot_velocity() -[qdot[0].item(), qdot[1].item(), qdot[2].item()],
                                      dcm_reactive_stepper.get_right_foot_velocity() - [qdot[0].item(), qdot[1].item(), qdot[2].item()]))
        try:
            dcm_force[0] = -dcm_force[0]
            dcm_force[1] = -dcm_force[1]
            dcm_force[2] = -dcm_force[2]
            if cnt_array[0] == cnt_array[1]:
                if is_left_leg_in_contact:
                    F[3:] = dcm_force[:3]
                else:
                    F[:3] = dcm_force[:3]
        except:
            F[:] = [0., 0., 0., 0., 0., 0.,]

        ##### Construct the input state for the policy #####

        stance_phase = dcm_reactive_stepper.get_contact_phase()
        contact_status, contact_forces = robot.end_effector_forces()

        if start_phase is None:
            start_phase = dcm_reactive_stepper.get_is_left_leg_in_contact()
        
        if start_phase == dcm_reactive_stepper.get_is_left_leg_in_contact() and start_step == False:
            start_step = True
            des_contact = prev_des_contact
        elif start_phase != dcm_reactive_stepper.get_is_left_leg_in_contact() and start_step == True:
            start_step = False
            des_contact = prev_des_contact
        time_mult = 1.0
        if start_step == True:
            time_mult = 1.0#2.0

        q_pos = q.copy()
        q_vel = qdot.copy()
        # Contact Conditioned Policy
        timeflight = time_mult * dcm_reactive_stepper.get_duration_of_flight_phase()
        dcm_opt_stance_phase_dur = time_mult * dcm_reactive_stepper.get_duration_of_stance_phase()
        time_rem_in_step = [dcm_opt_stance_phase_dur + timeflight - dcm_reactive_stepper.get_time_from_last_step_touchdown()]
        time_rem_in_next_step = [2.0*dcm_opt_stance_phase_dur + 2.0*timeflight - dcm_reactive_stepper.get_time_from_last_step_touchdown()]
        # Desired Velocity Conditioned Policy
        #v_des = v_des
        gait_idx = [-1] if gait == 'walk' else [1]
        # if prev_gait != gait:
        #     p.removeUserDebugItem(text_id)
        #     text_id = p.addUserDebugText(f"Gait: {gait}", [0, 0, 1], textColorRGB=[1, 0, 0], textSize=1.5)
        left = bolt_leg_ctrl.imp_ctrl_array[0]
        right = bolt_leg_ctrl.imp_ctrl_array[1]
        left_foot_location = left.pin_robot.data.oMf[left.frame_end_idx]
        right_foot_location = right.pin_robot.data.oMf[right.frame_end_idx]
        
        base_link_se3 = robot.pin_robot.data.oMf[robot.base_link_id]
        lf_base_link = left_foot_location.translation - base_link_se3.translation
        rf_base_link = right_foot_location.translation - base_link_se3.translation
        base_acc = robot.get_base_acceleration_world()

        shared_state = q_pos[2:].tolist() +\
                       q_vel.tolist() +\
                       lf_base_link.tolist() +\
                       rf_base_link.tolist() +\
                       stance_phase.tolist()
                    #    base_acc.tolist() +\
                    #    contact_forces[0][:3].tolist() +\
                    #    contact_forces[1][:3].tolist() +\
                    #    contact_status.tolist() #+\
        
        next_step_in_base = dcm_reactive_stepper.get_next_support_foot_position() - base_link_se3.translation 
        next_next_step_in_base = v_des[0]*(dcm_opt_stance_phase_dur + timeflight) + next_step_in_base
        
        contact_policy_input_state = shared_state +\
                                     next_step_in_base.tolist() +\
                                     time_rem_in_step
        contact_gait_policy_input_state = contact_policy_input_state +\
                                          gait_idx
        two_contact_policy_input_state = contact_policy_input_state +\
                                         next_next_step_in_base.tolist() +\
                                         time_rem_in_next_step
        two_contact_gait_policy_input_state = two_contact_policy_input_state +\
                                              gait_idx

        des_vel_policy_input_state = shared_state +\
                                     [v_des[0]] +\
                                     gait_idx +\
                                     time_rem_in_step

        ####################################################
        p_gain = 0.0001
        Kp = 2.0
        Kd = 0.1

        if apply_force_disturbance:
            if control_time >= t_force_start and control_time < t_force_end:
                p.applyExternalForce(objectUniqueId=robot.robotId, linkIndex=-1, forceObj=force,
                                posObj=[q[0], q[1], q[2]], flags=p.WORLD_FRAME)
            
        # Torques: model based and learned
        if learned_policy is not None and warmup <= i:
            if conditioning_type == "contact":
                policy_input_state = contact_policy_input_state
                policy_state = torch.tensor(policy_input_state, dtype=torch.float32)
                actions = learned_policy.infer(policy_state.reshape(1,-1))
                tau = actions.numpy()
            elif conditioning_type == "des_vel":
                policy_input_state = des_vel_policy_input_state
                policy_state = torch.tensor(policy_input_state, dtype=torch.float32)
                actions = learned_policy.infer(policy_state.reshape(1,-1))
                tau = actions.numpy()
            elif conditioning_type == "pd_contact":
                policy_input_state = contact_policy_input_state
                policy_state = torch.tensor(policy_input_state, dtype=torch.float32)
                actions = learned_policy.infer(policy_state.reshape(1,-1))
                q_next = actions.numpy()
                tau = Kp * (q_next[0] - q[7:]) - Kd * qdot[6:]
            elif conditioning_type == "pd_two_contact":
                policy_input_state = two_contact_policy_input_state
                policy_state = torch.tensor(policy_input_state, dtype=torch.float32)
                actions = learned_policy.infer(policy_state.reshape(1,-1))
                q_next = actions.numpy()
                tau = Kp * (q_next[0] - q[7:]) - Kd * qdot[6:]
            elif conditioning_type == "pd_two_contact_gait":
                policy_input_state = two_contact_gait_policy_input_state
                policy_state = torch.tensor(policy_input_state, dtype=torch.float32)
                actions = learned_policy.infer(policy_state.reshape(1,-1))
                q_next = actions.numpy()
                tau = Kp * (q_next[0] - q[7:]) - Kd * qdot[6:]
            elif conditioning_type == "pd_contact_gait":
                policy_input_state = contact_gait_policy_input_state
                policy_state = torch.tensor(policy_input_state, dtype=torch.float32)
                actions = learned_policy.infer(policy_state.reshape(1,-1))
                q_next = actions.numpy()
                tau = Kp * (q_next[0] - q[7:]) - Kd * qdot[6:]
            elif conditioning_type == "pd_des_vel":
                policy_input_state = des_vel_policy_input_state
                policy_state = torch.tensor(policy_input_state, dtype=torch.float32)
                actions = learned_policy.infer(policy_state.reshape(1,-1))
                q_next = actions.numpy()
                tau = Kp * (q_next[0] - q[7:]) - Kd * qdot[6:]
        else:
            tau = bolt_leg_ctrl.return_joint_torques(
                q.copy(),
                qdot.copy(),
                zero_cnt_gain(kp, cnt_array),
                zero_cnt_gain(kd, cnt_array),
                x_des_local,
                des_vel,
                F,
            )
        ## PD Actions
        if collect_learning_data:
            q_pd = q[7:] + (tau + Kd * qdot[6:]) / Kp

        # Record_data
        
        if failure_check(xd_com.tolist(), x_com.tolist()):
            batch_data['fails'].append([True])
            batch_data['time_to_fails'].append([control_time])
            print("\nBatchFailure")
            failed = True
            break
        
        if dcm_reactive_stepper.get_is_left_leg_in_contact():
            current_actual_contact = left_foot_location.translation
        else:
            current_actual_contact = right_foot_location.translation

        if collect_learning_data and control_time >= start_data_collection:
            batch_learning_data['contact_states'].append(contact_policy_input_state)
            batch_learning_data['two_contact_states'].append(two_contact_policy_input_state)
            batch_learning_data['contact_gait_states'].append(contact_gait_policy_input_state) 
            batch_learning_data['two_contact_gait_states'].append(two_contact_gait_policy_input_state)
            batch_learning_data['des_vel_states'].append(des_vel_policy_input_state)
            batch_learning_data['actions'].append(tau.tolist())
            batch_learning_data['pd_actions'].append(q_pd.tolist())

        batch_data['time'].append([control_time])
        batch_data['desired_gait'].append([gait])
        batch_data['desired_com_velocity'].append(v_des)
        batch_data['actual_com_velocity'].append(xd_com.tolist())
        batch_data['next_desired_contact'].append(dcm_reactive_stepper.get_next_support_foot_position().copy().tolist())
        batch_data['actual_com_position'].append(x_com.tolist())
        batch_data['torques'].append(tau.tolist())

        if not (stance_phase[0]==0 and stance_phase[1]==0):
            batch_data['current_desired_contact'].append(des_contact)
            batch_data['actual_contact'].append(current_actual_contact.tolist())

        prev_des_contact = dcm_reactive_stepper.get_next_support_foot_position().copy().tolist()
        prev_base_pos = q.copy().tolist()[:3]
        prev_gait = gait
        control_time += 0.001
        robot.send_joint_command(tau)
        p.stepSimulation()
    p.disconnect()
    if not failed:
        batch_data['fails'].append([False])
        batch_data['time_to_fails'].append([control_time])
    batch_data['total_distance'].append(total_distance)

def failure_check(com_vel, com_pos):
    fail = False
    # Velocity check
    if np.linalg.norm(com_vel) > 3.0:
        fail = True
    # Com position check
    if com_pos[2] < 0.15 or com_pos[2] > 0.5:
        fail = True

    return fail