import torch 
import torch.nn as nn

class FowardDynamicModel(nn.Module):
    def __init__(self, obs_dim, action_dim, output_dim):
        super(FowardDynamicModel, self).__init__()

        self.current_obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128,256),
            nn.LeakyReLU(),
            nn.Linear(256,256),
            nn.LeakyReLU(),
            nn.Linear(256,128)
        )

        self.action_encoder = nn.Sequential(

        )

    def foward(self, observation, action):
        observation = self.current_obs_encoder(observation)
        action = self.action_encoder(action)

        self