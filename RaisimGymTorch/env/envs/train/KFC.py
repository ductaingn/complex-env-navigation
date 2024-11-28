import torch
import torch.nn as nn
from RaisimGymTorch.algo.TD3 import TD3

class KFC(nn.Module):
    def __init__(self, pretrained_FDM:nn.Module, state_dim, action_dim, action_ub, discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2):
        super(KFC, self).__init__()

        self.FDM = pretrained_FDM
        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": action_ub,
            "discount": discount,
            "tau": tau,
        }

        kwargs["state_dim"] = state_dim
        kwargs["action_dim"] = action_dim
        kwargs["policy_noise"] = policy_noise*action_ub
        kwargs["noise_clip"] = noise_clip*action_ub
        kwargs["policy_freq"] = policy_freq

        self.TD3 = policy = TD3.TD3(**kwargs)


    def forward(self, x):
        return