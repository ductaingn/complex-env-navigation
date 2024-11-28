from RaisimGymTorch.algo.TD3 import TD3, utils
import numpy as np
import torch
import argparse
import os
from ruamel.yaml import YAML, dump, RoundTripDumper
from RaisimGymTorch.env.bin import train
from RaisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from RaisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver
from RaisimGymTorch.helper.utils_plot import plot_trajectory_prediction_result
import os
import math
import time
import RaisimGymTorch.algo.ppo.module as ppo_module
import torch.nn as nn
import numpy as np
import torch
from collections import Counter
import argparse
import pdb
import wandb
from RaisimGymTorch.env.envs.train.model import Forward_Dynamics_Model
from RaisimGymTorch.env.envs.train.trainer import FDM_trainer
from RaisimGymTorch.env.envs.train.action import UserCommand, Constant_command_sampler, Linear_time_correlated_command_sampler, Normal_time_correlated_command_sampler
from RaisimGymTorch.env.envs.train.storage import Buffer
import random
import io


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # task specification
    task_name = "FDM_train"

    # configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('-tw', '--tracking_weight', help='velocity command tracking policy weight path', type=str, default='')
    args = parser.parse_args()
    command_tracking_weight_path = args.tracking_weight
    if command_tracking_weight_path == '':
        command_tracking_weight_path = 'data/command_tracking_flat/testing_2_layers/full_0.pt'
    # check if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # directories
    task_path = os.path.dirname(os.path.realpath(__file__))
    home_path = task_path + "/../../../.."

    # config
    cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))
    cfg["environment"]["determine_env"] = 0
    cfg["environment"]["evaluate"] = False
    cfg["environment"]["random_initialize"] = True
    cfg["environment"]["point_goal_initialize"] = False
    cfg["environment"]["CVAE_data_collection_initialize"] = False
    cfg["environment"]["safe_control_initialize"] = False
    cfg["environment"]["CVAE_environment_initialize"] = False

    # user command sampling
    user_command = UserCommand(cfg, cfg['environment']['num_envs'])
    command_sampler_constant = Constant_command_sampler(user_command)
    command_sampler_linear_correlated = Linear_time_correlated_command_sampler(user_command,
                                                                            beta=cfg["data_collection"]["linear_time_correlated_command_sampler_beta"])
    command_sampler_normal_correlated = Normal_time_correlated_command_sampler(user_command, cfg["environment"]["command"],
                                                                            sigma=cfg["data_collection"]["normal_time_correlated_command_sampler_sigma"],
                                                                            std_scale_fixed=False)

    # create environment from the configuration file
    yaml = YAML()
    stream = io.StringIO()
    yaml.dump(cfg['environment'], stream)
    environment_cfg = stream.getvalue()

    env = VecEnv(train.RaisimGymEnv(home_path + "/rsc", environment_cfg), cfg['environment'], normalize_ob=False)

    # shortcuts
    user_command_dim = 3
    proprioceptive_sensor_dim = 81
    lidar_dim = 360
    assert env.num_obs == proprioceptive_sensor_dim + lidar_dim, "Check configured sensor dimension"

    # training rollout config
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    command_period_steps = math.floor(cfg['data_collection']['command_period'] / cfg['environment']['control_dt'])
    total_steps = n_steps * env.num_envs
    assert n_steps % command_period_steps == 0, "Total steps in training should be divided by command period steps."

    state_dim = cfg["architecture"]["state_encoder"]["input"]
    command_dim = cfg["architecture"]["command_encoder"]["input"]
    P_col_dim = cfg["architecture"]["traj_predictor"]["collision"]["output"]
    coordinate_dim = cfg["architecture"]["traj_predictor"]["coordinate"]["output"]   # Just predict x, y coordinate (not yaw)

    # use naive concatenation for encoding COM vel history
    COM_feature_dim = cfg["architecture"]["COM_encoder"]["naive"]["input"]
    COM_history_time_step = cfg["architecture"]["COM_encoder"]["naive"]["time_step"]
    COM_history_update_period = int(cfg["architecture"]["COM_encoder"]["naive"]["update_period"] / cfg["environment"]["control_dt"])
    assert state_dim - lidar_dim == COM_feature_dim * COM_history_time_step, "Check COM_encoder output and state_encoder input in the cfg.yaml"

    command_tracking_ob_dim = user_command_dim + proprioceptive_sensor_dim
    command_tracking_act_dim = env.num_acts

    COM_buffer = Buffer(env.num_envs, COM_history_time_step, COM_feature_dim)

    FDM_model = Forward_Dynamics_Model(state_encoding_config=cfg["architecture"]["state_encoder"],
                                            command_encoding_config=cfg["architecture"]["command_encoder"],
                                            recurrence_config=cfg["architecture"]["recurrence"],
                                            prediction_config=cfg["architecture"]["traj_predictor"],
                                            device=device)


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    policy = TD3.TD3(**kwargs)

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")
