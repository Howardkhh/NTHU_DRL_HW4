from collections import deque
import numpy as np
import torch
from osim.env import L2M2019Env

class Network(torch.nn.Module):
    def __init__(self, input_shape, n_actions, action_space):
        super(Network, self).__init__()
        self.meanNet = torch.nn.Sequential(
            torch.nn.Linear(input_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_actions)
        )
        self.logNet = torch.nn.Sequential(
            torch.nn.Linear(input_shape, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_actions)
        )

        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2, dtype=torch.float32)
        self.action_bias = torch.tensor((action_space.high + action_space.low) / 2, dtype=torch.float32)

    def forward(self, state: torch.Tensor):
        mean = self.meanNet(state)
        log_std = self.logNet(state)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def sample(self, state: torch.Tensor):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        sample = normal.rsample()
        squashed_sample = torch.tanh(sample)
        action = squashed_sample * self.action_scale + self.action_bias
        log_prob = normal.log_prob(sample) - torch.log(self.action_scale * (1 - squashed_sample.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        deterministic_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic_action
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class Agent():
    def __init__(self):
        env = L2M2019Env(visualize=False, difficulty=2)
        self.behavior_network = Network(env.observation_space.shape[0], env.action_space.shape[0], env.action_space)
        self.behavior_network.load_state_dict(torch.load("109062102_hw4_data", map_location="cpu"))
        self.behavior_network.eval()
        self.counter = 0
        self.last_action = None
        self.frames = deque([], maxlen=4)

    def preprocess(self, state):
        def extract_values(obj):
            if isinstance(obj, dict):
                for value in obj.values():
                    yield from extract_values(value)
            elif isinstance(obj, list):
                for item in obj:
                    yield from extract_values(item)
            elif isinstance(obj, np.ndarray):
                for item in obj.flatten():
                    yield from extract_values(item)
            else:
                yield obj

        # Create a 1D list of all values
        values = list(extract_values(state))
        
        # Convert the list to a numpy array and then to a torch tensor
        return torch.tensor([values], dtype=torch.float32, device="cpu")

    def act(self, observation):
        # if self.counter % 4 == 0:
            # self.last_action = self.behavior_network.sample(self.preprocess(observation))[2]
        # self.counter += 1
        self.last_action = self.behavior_network.sample(self.preprocess(observation))[2]
        return self.last_action.detach().cpu().numpy()[0]