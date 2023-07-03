from typing import Tuple

import torch

from sheeprl.models.models import MLP, NatureCNN


class MuzeroAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.representation: torch.nn.Module
        self.prediction: torch.nn.Module
        self.dynamics: torch.nn.Module
        self.training_steps: int = 0

    def initial_inference(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)
        return hidden_state.unsqueeze(0), policy_logits, value

    def recurrent_inference(
        self, action: torch.Tensor, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reward, next_hidden_state = self.dynamics(action, hidden_state)
        policy_logits, value = self.prediction(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value

    def forward(self, action, hidden_state):
        return self.recurrent_inference(action, hidden_state)

    @torch.no_grad()
    def gradient_norm(self):
        """Compute the norm of the parameters' gradients."""
        total_norm = 0
        p_with_grads = [p for p in self.parameters() if p.grad is not None]
        if not p_with_grads:
            raise RuntimeError("No parameters have gradients. Run the backward method first.")
        for p in p_with_grads:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm**0.5


class GruMlpDynamics(torch.nn.Module):
    def __init__(self, hidden_state_size=256):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=1, hidden_size=hidden_state_size)
        self.mlp = MLP(input_dims=hidden_state_size, output_dim=1)

    def forward(self, x, h0):
        y, h1 = self.gru(x, h0)
        y = self.mlp(y)
        return y, h1


class Predictor(torch.nn.Module):
    def __init__(self, hidden_state_size=256, num_actions=4):
        super().__init__()
        self.mlp_actor1 = torch.nn.Linear(in_features=hidden_state_size, out_features=16)
        self.act_actor = torch.nn.ELU()
        self.mlp_actor2 = torch.nn.Linear(in_features=16, out_features=num_actions)

        self.mlp_value1 = torch.nn.Linear(in_features=hidden_state_size, out_features=16)
        self.act_value = torch.nn.ELU()
        self.mlp_value2 = torch.nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        policy = self.mlp_actor2(self.act_actor(self.mlp_actor1(x)))
        value = self.mlp_value2(self.act_value(self.mlp_value1(x)))
        return policy, value


class RecurrentMuzero(MuzeroAgent):
    def __init__(self, hidden_state_size=256, num_actions=4):
        super().__init__()
        self.representation = NatureCNN(in_channels=3, features_dim=hidden_state_size)
        self.dynamics: torch.nn.Module = GruMlpDynamics(hidden_state_size=hidden_state_size)
        self.prediction: torch.nn.Module = Predictor(hidden_state_size=hidden_state_size, num_actions=num_actions)


if __name__ == "__main__":
    batch_size = 32
    sequence_len = 5
    agent = RecurrentMuzero()
    # Player:
    print("Player:")
    observation = torch.rand(1, 3, 64, 64)
    hidden_state, policy_logits, value = agent.initial_inference(observation)
    print("Initial inference:")
    print(hidden_state.shape)
    print(policy_logits.shape)
    print(value.shape)

    action = torch.rand(1, 1, 1)
    hidden_state, policy_logits, reward, value = agent.recurrent_inference(action, hidden_state)
    print("Recurrent inference:")
    print(hidden_state.shape)
    print(policy_logits.shape)
    print(reward.shape)
    print(value.shape)

    ## Trainer:
    print("Trainer:")
    observation = torch.rand(batch_size, 3, 64, 64)
    hidden_state, policy_logits, value = agent.initial_inference(observation)
    print("Initial inference:")
    print(hidden_state.shape)
    print(policy_logits.shape)
    print(value.shape)

    action = torch.rand(1, batch_size, 1)
    hidden_state, policy_logits, reward, value = agent.recurrent_inference(action, hidden_state)
    print("Recurrent inference:")
    print(hidden_state.shape)
    print(policy_logits.shape)
    print(reward.shape)
    print(value.shape)
    # action2 = torch.randint(0, 4, (1, 1)).to(torch.float32)
    # last_hidden_state, reward, policy_logits, value = agent.recurrent_inference(action2, next_hidden_state)
    # print(last_hidden_state.shape)
    # print(reward.shape)
    # print(policy_logits.shape)
    # print(value.shape)