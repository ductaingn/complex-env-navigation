from ruamel.yaml import YAML, dump, RoundTripDumper
from RaisimGymTorch.env.bin import test
from RaisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import os
import math
import time
import RaisimGymTorch.algo.ppo.module as ppo_module
import torch.nn as nn
import numpy as np
import torch
import argparse
import pdb
from RaisimGymTorch.env.envs.train.model import Forward_Dynamics_Model
from RaisimGymTorch.env.envs.train.action import UserCommand, Stochastic_action_planner_uniform_bin, Stochastic_action_planner_w_ITS
from RaisimGymTorch.env.envs.train.model import Informed_Trajectory_Sampler_inference
from RaisimGymTorch.env.envs.train.storage import Buffer
import random
from RaisimGymTorch.env.envs.train.global_planner import Analytic_planner
from fastdtw import fastdtw
from dtw import dtw
from scipy.spatial.distance import euclidean
from joblib import Parallel, delayed
import io

"""
Point-Goal Navigation using Time-correlated random sampler & Informed Trajectory Sampler (ITS)

- below code simply merge samples from two different sampler and optimize using MPPI
- set number of samples from Time-correlated random sampler to 0 if you only want to sample using ITS
 
"""

def transform_coordinate_LW(w_init_coordinate, l_coordinate_traj):
    """
    Transform LOCAL frame coordinate trajectory to WORLD frame coordinate trajectory
    (LOCAL frame --> WORLD frame)

    :param w_init_coordinate: initial coordinate in WORLD frame (1, coordinate_dim)
    :param l_coordinate_traj: coordintate trajectory in LOCAL frame (n_step, coordinate_dim)
    :return:
    """
    transition_matrix = np.array([[np.cos(w_init_coordinate[0, 2]), np.sin(w_init_coordinate[0, 2])],
                                  [- np.sin(w_init_coordinate[0, 2]), np.cos(w_init_coordinate[0, 2])]], dtype=np.float32)
    w_coordinate_traj = np.matmul(l_coordinate_traj, transition_matrix)
    w_coordinate_traj += w_init_coordinate[:, :-1]
    return w_coordinate_traj

def transform_coordinate_WL(w_init_coordinate, w_coordinate_traj):
    """
    Transform WORLD frame coordinate trajectory to LOCAL frame coordinate trajectory
    (WORLD frame --> LOCAL frame)

    :param w_init_coordinate: initial coordinate in WORLD frame (1, coordinate_dim)
    :param w_coordinate_traj: coordintate trajectory in WORLD frame (n_step, coordinate_dim)
    :return:
        l_coordinate_traj: coordintate trajectory in LOCAL frame (n_step, coordinate_dim)
    """
    transition_matrix = np.array([[np.cos(w_init_coordinate[0, 2]), np.sin(w_init_coordinate[0, 2])],
                                  [- np.sin(w_init_coordinate[0, 2]), np.cos(w_init_coordinate[0, 2])]], dtype=np.float32)
    l_coordinate_traj = w_coordinate_traj - w_init_coordinate[:, :-1]
    l_coordinate_traj = np.matmul(l_coordinate_traj, transition_matrix.T)
    return l_coordinate_traj

def compute_dtw(x, y, type="fast"):
    if type == "normal":
        dtw_distance = dtw(x, y, dist_method=euclidean).distance
        normalize_factor = (x.shape[0] + y.shape[0]) / 2
        return dtw_distance / normalize_factor
    elif type == "not_normal":
        dtw_distance = dtw(x, y, dist_method=euclidean).distance
        return dtw_distance
    else:
        return fastdtw(x, y, dist=euclidean)

# task specification
task_name = "point_goal_nav"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-fw', '--fdm_weight', help='Forward Dynamics Model weight path', type=str, default='')
parser.add_argument('-iw', '--its_weight', help='Informed Trajectory Sampler weight path', type=str, default='')
parser.add_argument('-tw', '--tracking_weight', help='velocity command tracking policy weight path', type=str, default='')
parser.add_argument('-v', '--validate', help='validation or test', type=bool, default=False)
parser.add_argument('-cf', '--cf', help='extra note', type=str, default='')
args = parser.parse_args()
FDM_weight_path = args.fdm_weight
ITS_weight_path = args.its_weight
command_tracking_weight_path = args.tracking_weight
validation = args.validate
extra_note = args.cf

evaluate_seed = 10
random.seed(evaluate_seed)
np.random.seed(evaluate_seed)
torch.manual_seed(evaluate_seed)

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# complete path configuration
if FDM_weight_path == '':
    assert cfg["path"]["home"] != '' or cfg["path"]["FDM"] != '', "Weight path configuration not complete."
    FDM_weight_path = cfg["path"]["home"] + cfg["path"]["FDM"]
if ITS_weight_path == '':
    assert cfg["path"]["home"] != '' or cfg["path"]["ITS"] != '', "Weight path configuration not complete."
    ITS_weight_path = cfg["path"]["home"] + cfg["path"]["ITS"]
if command_tracking_weight_path == '':
    assert cfg["path"]["home"] != '' or cfg["path"]["command_tracking"] != '', "Weight path configuration not complete."
    command_tracking_weight_path = cfg["path"]["home"] + cfg["path"]["command_tracking"]

# load model architecture
cfg["environment"]["test_initialize"]["point_goal"] = True
cfg["environment"]["test_initialize"]["safety_control"] = False

# user command sampling
user_command = UserCommand(cfg, cfg['CVAE_path_track']['planner']['wo_CVAE_number_of_sample'])

# create environment from the configuration file
cfg['environment']['num_envs'] = 1

# create environment from the configuration file
yaml = YAML()
stream = io.StringIO()
yaml.dump(cfg['environment'], stream)
environment_cfg = stream.getvalue()

env = VecEnv(test.RaisimGymEnv(home_path + "/rsc", environment_cfg), cfg['environment'], normalize_ob=False)

# shortcuts
user_command_dim = 3
proprioceptive_sensor_dim = 81
lidar_dim = 360
state_dim = cfg["environment_model"]["architecture"]["state_encoder"]["input"]
command_period_steps = math.floor(cfg['command_tracking']['command_period'] / cfg['environment']['control_dt'])
assert env.num_obs == proprioceptive_sensor_dim + lidar_dim, "Check configured sensor dimension"

# Use naive concatenation for encoding COM vel history
COM_feature_dim = cfg["environment_model"]["architecture"]["COM_encoder"]["naive"]["input"]
COM_history_time_step = cfg["environment_model"]["architecture"]["COM_encoder"]["naive"]["time_step"]
COM_history_update_period = int(cfg["environment_model"]["architecture"]["COM_encoder"]["naive"]["update_period"] / cfg["environment"]["control_dt"])
assert state_dim - lidar_dim == COM_feature_dim * COM_history_time_step, "Check COM_encoder output and state_encoder input in the cfg.yaml"

command_tracking_ob_dim = user_command_dim + proprioceptive_sensor_dim
command_tracking_act_dim = env.num_acts

COM_buffer = Buffer(env.num_envs, COM_history_time_step, COM_feature_dim)

# Load pre-trained command tracking policy weight
assert command_tracking_weight_path != '', "Pre-trained command tracking policy weight path should be determined."
# command_tracking_policy = ppo_module.MLP(cfg['command_tracking']['architecture'], nn.LeakyReLU,
                                        #  command_tracking_ob_dim, command_tracking_act_dim)
command_tracking_policy = nn.Sequential(nn.Linear(84,128,), nn.LeakyReLU(), nn.Linear(128,32), nn.LeakyReLU(), nn.Linear(32,4))
command_tracking_policy.load_state_dict(torch.load(command_tracking_weight_path, map_location=device))
command_tracking_policy.to(device)
command_tracking_weight_dir = command_tracking_weight_path.rsplit('/', 1)[0] + '/'
iteration_number = command_tracking_weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
env.load_scaling(command_tracking_weight_dir, int(iteration_number))

print("Loaded command tracking policy weight from {}\n".format(command_tracking_weight_path))

# Load learned environment model weight
loaded_environment_model = Forward_Dynamics_Model(state_encoding_config=cfg["environment_model"]["architecture"]["state_encoder"],
                                                  command_encoding_config=cfg["environment_model"]["architecture"]["command_encoder"],
                                                  recurrence_config=cfg["environment_model"]["architecture"]["recurrence"],
                                                  prediction_config=cfg["environment_model"]["architecture"]["traj_predictor"],
                                                  device=device)
loaded_environment_model.load_state_dict(torch.load(FDM_weight_path, map_location=device)['model_architecture_state_dict'])
loaded_environment_model.eval()
loaded_environment_model.to(device)
print("Loaded Forward Dynamics Model weight from {}\n".format(FDM_weight_path))

# Load sampler
n_prediction_step = int(cfg["CVAE_path_track"]["planner"]["prediction_period"] / cfg['command_tracking']['command_period'])
wo_cvae_sampler = Stochastic_action_planner_uniform_bin(command_range=cfg["environment"]["command"],
                                                        n_sample=cfg["CVAE_path_track"]["planner"]["wo_CVAE_number_of_sample"],
                                                        n_horizon=n_prediction_step,
                                                        n_bin=cfg["CVAE_path_track"]["planner"]["number_of_bin"],
                                                        beta=cfg["CVAE_path_track"]["planner"]["wo_CVAE_beta"],
                                                        gamma=cfg["CVAE_path_track"]["planner"]["gamma"],
                                                        sigma=cfg["CVAE_path_track"]["planner"]["sigma"],
                                                        noise_sigma=0.1,
                                                        noise=False,
                                                        action_dim=user_command_dim,
                                                        random_command_sampler=user_command)

w_cvae_sampler = Informed_Trajectory_Sampler_inference(
    latent_dim=cfg["CVAE_path_track"]["architecture"]["latent_dim"],
    state_encoding_config=cfg["CVAE_path_track"]["architecture"]["state_encoder"],
    waypoint_encoding_config=cfg["CVAE_path_track"]["architecture"]["waypoint_encoder"],
    waypoint_recurrence_encoding_config=cfg["CVAE_path_track"]["architecture"]["waypoint_recurrence_encoder"],
    latent_decoding_config=cfg["CVAE_path_track"]["architecture"]["latent_decoder"],
    recurrence_decoding_config=cfg["CVAE_path_track"]["architecture"]["recurrence_decoder"],
    command_decoding_config=cfg["CVAE_path_track"]["architecture"]["command_decoder"],
    device=device,
    trained_weight=ITS_weight_path,
    cfg_command=cfg["environment"]["command"]
)
w_cvae_sampler.eval()
w_cvae_sampler.to(device)
print("Loaded Informed Trajectory Sampler weight from {}\n".format(ITS_weight_path))

# Set action planner
action_planner = Stochastic_action_planner_w_ITS(wo_cvae_sampler=wo_cvae_sampler,
                                                 w_cvae_sampler=w_cvae_sampler,
                                                 wo_cvae_n_sample=cfg["CVAE_path_track"]["planner"]["wo_CVAE_number_of_sample"],
                                                 w_cvae_n_sample=cfg["CVAE_path_track"]["planner"]["CVAE_number_of_sample"],
                                                 n_prediction_step=n_prediction_step,
                                                 gamma=cfg["CVAE_path_track"]["planner"]["gamma"],
                                                 beta=cfg["CVAE_path_track"]["planner"]["CVAE_beta"])

# MUST safe period from collision
MUST_safety_period = 3.0
MUST_safety_period_n_steps = int(MUST_safety_period / cfg['command_tracking']['command_period'])

# Set constant
collision_threshold = 0.3
goal_distance_threshold = 10
num_goals = cfg["environment"]["n_goals_per_env"]
if validation:
    init_seed = cfg["environment"]["seed"]["validate"]
    print("Validating ...")
else:
    init_seed = cfg["environment"]["seed"]["evaluate"]
    print("Evaluating ...")
goal_time_limit = 180.

# hyperparameter for path tracking
lookahead_path_length = 4.8
nDTX_temperature = 8

# Make directory to save results
num_cvae_sample = cfg["CVAE_path_track"]["planner"]["CVAE_number_of_sample"]
num_wo_cvae_sample = cfg["CVAE_path_track"]["planner"]["wo_CVAE_number_of_sample"]

# Empty container to check time
time_check = []

print("<<-- Evaluating Point Goal Navigation -->>")

# pdb.set_trace()

for grid_size in [2.3, 3., 4., 5.]:
    print("===========================================")
    print(f"Grid_{str(grid_size)}:")
    print("===========================================")

    # Set obstacle grid size
    cfg["environment"]["test_obstacle_grid_size"] = grid_size

    for env_id in range(cfg["environment"]["n_evaluate_envs"]):
        # Generate new environment with different seed (reset is automatically called)
        cfg["environment"]["seed"]["evaluate"] = env_id * 10 + init_seed
        yaml = YAML()
        stream = io.StringIO()
        yaml.dump(cfg['environment'], stream)
        environment_cfg = stream.getvalue()
        env = VecEnv(test.RaisimGymEnv(home_path + "/rsc", environment_cfg), cfg['environment'], normalize_ob=False)
        env.load_scaling(command_tracking_weight_dir, int(iteration_number))

        # set analytic planner
        map_size = np.zeros(2, dtype=np.float32)
        env.get_map_size(map_size)
        assert map_size[0] == map_size[1], "Map should be a square"
        analytic_planner = Analytic_planner(env, map_size[0], cfg["environment"]["analytic_planner"]["planning_time"], seed=evaluate_seed)

        n_test_case = 0
        n_path_failure = 0  # Count number of finding path failure

        while n_test_case < num_goals:
            # Reset
            env.initialize_n_step()
            env.reset()
            action_planner.reset()
            goal_position = env.set_goal()[np.newaxis, :]
            print(n_path_failure)
            COM_buffer.reset()

            # Initialize
            step = 0
            sample_user_command = np.zeros(3)
            goal_current_duration = 0.
            traversal_distance = 0.
            previous_coordinate = None
            current_coordinate = None

            # plan path with analytic planner
            start_position = env.coordinate_observe()[0, :2]
            planned_path = analytic_planner.plan(start_position.astype(np.float64), goal_position[0, :].astype(np.float64))

            if cfg["environment"]["render"] and cfg["environment"]["analytic_planner"]["visualize"] and planned_path is not None:
                analytic_planner.visualize_path()

            if planned_path is None:
                n_path_failure += 1
                if n_path_failure > 10:
                    break
            else:
                # Available test case found
                n_test_case += 1

                # Save COM trajectory to compute path similarity with DTW
                traversed_path = []

                while True:
                    frame_start = time.time()
                    control_start = time.time()

                    new_action_time = step % command_period_steps == 0

                    # log traversal distance (just env 0)
                    previous_coordinate = current_coordinate
                    current_coordinate = env.coordinate_observe()
                    if previous_coordinate is not None:
                        delta_coordinate = current_coordinate[0, :-1] - previous_coordinate[0, :-1]
                        traversal_distance += (delta_coordinate[0] ** 2 + delta_coordinate[1] ** 2) ** 0.5

                    # observation before taking step
                    obs, _ = env.observe(False)

                    # update COM feature
                    if step % COM_history_update_period == 0:
                        COM_feature = np.concatenate((obs[:, :3], obs[:, 15:21]), axis=1)
                        COM_buffer.update(COM_feature)

                    if new_action_time:
                        if step != 0:
                            # Pass the one-step planned command to the low-level controller
                            sample_user_command = sample_user_command_traj[1, :]

                        # plan_start = time.time()

                        # prepare state
                        init_coordinate_obs = env.coordinate_observe()
                        lidar_data = obs[0, proprioceptive_sensor_dim:]
                        COM_history_feature = COM_buffer.return_data(flatten=True)[0, :]
                        state = np.concatenate((lidar_data, COM_history_feature)).astype(np.float32)

                        # compute final goal position w/ respect to local frame
                        goal_position_L = transform_coordinate_WL(init_coordinate_obs, goal_position)
                        current_goal_distance = np.sqrt(np.sum(np.power(goal_position_L, 2)))
                        goal_position_L *= np.clip(goal_distance_threshold / current_goal_distance, a_min=None, a_max=1.)
                        goal_position_L = goal_position_L.astype(np.float32)
                        current_goal_distance = np.sqrt(np.sum(np.power(goal_position_L, 2)))

                        # use path similarity as reward for path tracking
                        waypoints_list_L = transform_coordinate_WL(init_coordinate_obs, planned_path)
                        waypoint_distance_list = np.sqrt(np.sum(np.power(waypoints_list_L, 2), axis=-1))
                        closest_waypoint_idx = np.argmin(waypoint_distance_list)
                        waypoints_list_L_closest = waypoints_list_L - waypoints_list_L[closest_waypoint_idx]
                        along_waypoint_distance = 0
                        farthest_waypoint_idx = planned_path.shape[0] - 1
                        for idx in range(closest_waypoint_idx, waypoints_list_L_closest.shape[0] - 1):
                            along_waypoint_distance += np.linalg.norm(waypoints_list_L_closest[idx + 1] - waypoints_list_L_closest[idx], ord=2)
                            if along_waypoint_distance > lookahead_path_length:
                                farthest_waypoint_idx = idx + 1
                                break

                        waypoint_idx = list(range(closest_waypoint_idx, farthest_waypoint_idx + 1))
                        waypoints_list_L = waypoints_list_L[waypoint_idx]  # (n_waypoint, 2)

                        if cfg["environment"]["render"] and cfg["environment"]["analytic_planner"]["visualize"]:
                            env.visualize_waypoints(planned_path[[closest_waypoint_idx, farthest_waypoint_idx]])

                        # interpolate for sparsely sampled points
                        if farthest_waypoint_idx - closest_waypoint_idx < 4 and along_waypoint_distance > lookahead_path_length:
                            if farthest_waypoint_idx - closest_waypoint_idx == 3:
                                interpolated_waypoints = []
                                for idx in range(farthest_waypoint_idx - closest_waypoint_idx):
                                    interpolated_waypoints.append(np.linspace(waypoints_list_L[idx], waypoints_list_L[idx + 1], num=3)[:-1, :])
                                interpolated_waypoints.append(waypoints_list_L[-1, :][np.newaxis, :])
                                waypoints_list_L = np.concatenate(interpolated_waypoints, axis=0)
                            elif farthest_waypoint_idx - closest_waypoint_idx == 2:
                                interpolated_waypoints = []
                                for idx in range(farthest_waypoint_idx - closest_waypoint_idx):
                                    interpolated_waypoints.append(np.linspace(waypoints_list_L[idx], waypoints_list_L[idx + 1], num=4)[:-1, :])
                                interpolated_waypoints.append(waypoints_list_L[-1, :][np.newaxis, :])
                                waypoints_list_L = np.concatenate(interpolated_waypoints, axis=0)
                            elif farthest_waypoint_idx - closest_waypoint_idx == 1:
                                waypoints_list_L = np.linspace(waypoints_list_L[0], waypoints_list_L[1], num=8)

                        # sample command sequences
                        # one-step lookahead planning (first step command is already determined)
                        action_candidates = action_planner.sample(observation=torch.from_numpy(state).unsqueeze(0).to(device),
                                                                  goal_position=None,
                                                                  waypoints=torch.from_numpy(waypoints_list_L[:, np.newaxis, :]).to(device),
                                                                  current_command=None)
                        action_candidates[0, :, :] = sample_user_command
                        action_planner.sampled_command_traj = action_candidates

                        # simulate sampled command sequences
                        state = np.tile(state, (num_wo_cvae_sample + num_cvae_sample, 1))
                        predicted_P_cols, predicted_coordinates = loaded_environment_model(torch.from_numpy(state).to(device),
                                                                                           torch.from_numpy(action_candidates).to(device),
                                                                                           training=False)
                        predicted_P_cols = np.squeeze(predicted_P_cols, axis=-1)

                        # Hard constraint for collision
                        for sample_id in range(num_wo_cvae_sample + num_cvae_sample):
                            current_done = np.where(predicted_P_cols[:, sample_id] > collision_threshold)[0]
                            if len(current_done) != 0:
                                done_idx = np.min(current_done)
                                predicted_coordinates[done_idx + 1:, sample_id, :] = predicted_coordinates[done_idx, sample_id, :]
                                predicted_P_cols[done_idx + 1:, sample_id] = predicted_P_cols[done_idx, sample_id]

                        # compute reward (goal reward + safety reward)
                        if farthest_waypoint_idx - closest_waypoint_idx < 4 and along_waypoint_distance < lookahead_path_length:
                            # if the robot is very near the goal, give reward directly with goal position
                            delta_goal_distance = current_goal_distance - np.sqrt(np.sum(np.power(predicted_coordinates - goal_position_L, 2), axis=-1))
                            goal_reward = np.sum(delta_goal_distance, axis=0)
                        else:
                            padded_predicted_coordinates = np.concatenate((np.zeros((1, predicted_coordinates.shape[1], predicted_coordinates.shape[2])), predicted_coordinates), axis=0)
                            results = Parallel(n_jobs=8)(delayed(compute_dtw)(waypoints_list_L, padded_predicted_coordinates[:, i, :]) for i in range(num_wo_cvae_sample + num_cvae_sample))  # takes about 0.1 [s]
                            dtw_distance = np.array([r[0] for r in results])
                            dtw_distance -= np.min(dtw_distance)
                            goal_reward = np.exp(- dtw_distance / nDTX_temperature)
                            goal_reward = np.array(goal_reward)

                        goal_reward -= np.min(goal_reward)
                        goal_reward /= (np.max(goal_reward) + 1e-5)  # normalize reward

                        safety_reward = 1 - predicted_P_cols
                        safety_reward = np.mean(safety_reward, axis=0)
                        safety_reward /= (np.max(safety_reward) + 1e-5)  # normalize reward

                        reward = 1.0 * goal_reward + 1.0 * safety_reward

                        # exclude trajectory that collides with obstacle
                        coll_idx = np.where(np.sum(np.where(predicted_P_cols[:MUST_safety_period_n_steps, :] > collision_threshold, 1, 0), axis=0) != 0)[0]
                        if len(coll_idx) != (num_wo_cvae_sample + num_cvae_sample):
                            reward[coll_idx] = 0

                        # optimize command sequence
                        _, sample_user_command_traj = action_planner.action(reward)

                        # simulate optimized command sequence
                        state = state[0, :][np.newaxis, :]
                        predicted_P_cols, predicted_coordinates = loaded_environment_model(torch.from_numpy(state).to(device),                                                            torch.from_numpy(sample_user_command_traj[:, np.newaxis, :]).to(device),training=False)

                        # Hard constraint for collision
                        predicted_P_cols = np.squeeze(predicted_P_cols, axis=-1)
                        current_done = np.where(predicted_P_cols[:, 0] > collision_threshold)[0]
                        if len(current_done) != 0:
                            done_idx = np.min(current_done)
                            predicted_coordinates[done_idx + 1:, 0, :] = predicted_coordinates[done_idx, 0, :]
                            predicted_P_cols[done_idx + 1:, 0] = predicted_P_cols[done_idx, 0]

                        # visualize predicted modified command trajectory
                        if cfg["environment"]["render"]:
                            w_coordinate_modified_command_path = transform_coordinate_LW(init_coordinate_obs, predicted_coordinates[:, 0, :])
                            P_col_modified_command_path = predicted_P_cols[:, 0][:, np.newaxis]
                            env.visualize_modified_command_traj(w_coordinate_modified_command_path,
                                                                P_col_modified_command_path,
                                                                collision_threshold)

                        # Save COM coordinate
                        traversed_path.append(init_coordinate_obs[0, :-1])  # (coordinate_dim,)

                        # plan_end = time.time()

                        # print(plan_end - plan_start)

                    # Execute first command in optimized command sequence using command tracking controller
                    tracking_obs = np.concatenate((sample_user_command, obs[0, :proprioceptive_sensor_dim]))[np.newaxis, :]
                    tracking_obs = env.force_normalize_observation(tracking_obs, type=1)
                    tracking_obs = tracking_obs.astype(np.float32)

                    with torch.no_grad():
                        tracking_action = command_tracking_policy(torch.from_numpy(tracking_obs).to(device))

                    control_end = time.time()

                    _, done = env.step(tracking_action.cpu().detach().numpy())

                    # Update progress
                    step += 1
                    goal_current_duration += cfg['environment']['control_dt']

                    frame_end = time.time()

                    # # Check time
                    # if new_action_time:
                    #     time_check.append(control_end - control_start)
                    #     if len(time_check) == 5000:
                    #         time_check = np.array(time_check)
                    #         print(f"Mean: {np.mean(time_check[50:])}")
                    #         print(f"Std: {np.std(time_check[50:])}")
                    #         pdb.set_trace()

                    # if new_action_time:
                    #    print(control_end - control_start)

                    # if new_action_time:
                    #     print(frame_end - frame_start)

                    if cfg["realistic"]:
                        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                        if wait_time > 0.:
                            time.sleep(wait_time)

                    if goal_current_duration > goal_time_limit:
                        done[0] = True

                    # fail
                    if done[0] == True:
                        break
                    # success
                    elif current_goal_distance < 0.6:
                        if cfg["environment"]["render"] and cfg["environment"]["visualize_path"]:
                            pdb.set_trace()

                        # Update
                        traversed_path = np.stack(traversed_path, axis=0)
                        total_normal_dtw = compute_dtw(planned_path, traversed_path, type="normal")

                        print(f"Time: {round(step * cfg['environment']['control_dt'], 1)}")
                        print(f"Distance: {round(traversal_distance, 1)}")
                        print(f"Normal DTW: {round(total_normal_dtw, 1)}")
                        print("-------------------------------")

                        break

# env.stop_video_recording()
env.turn_off_visualization()