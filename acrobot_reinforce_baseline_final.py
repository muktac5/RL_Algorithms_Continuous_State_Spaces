import os
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import imageio

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, 128)
        self.hidden_layer2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        x = torch.softmax(self.output_layer(x), dim=-1)
        return x


class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, 128)
        self.hidden_layer2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x


def reinforce_with_baseline(env, policy_net, value_net, num_episodes, gamma, alpha_policy, alpha_value, seed, reward_scale=1.0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=alpha_policy)
    value_optimizer = optim.Adam(value_net.parameters(), lr=alpha_value)
    
    rewards_per_episode = []
    value_loss_history = []
    policy_loss_history = []
    steps_per_episode = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_rewards, log_probs, values = [], [], []
        total_steps = 0
        
        while True:
            state_tensor = torch.FloatTensor(state)
            action_probs = policy_net(state_tensor)
            state_value = value_net(state_tensor)
            
            action = torch.distributions.Categorical(action_probs).sample()
            next_state, reward, done, truncated, _ = env.step(action.item())
            
            log_probs.append(torch.log(action_probs[action]))
            values.append(state_value)
            episode_rewards.append(reward * reward_scale)
            total_steps += 1
            
            if done or truncated:
                break
            state = next_state
        
        G = 0
        returns = []
        for reward in reversed(episode_rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        values_tensor = torch.stack(values).squeeze()
        advantage = returns - values_tensor.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
        policy_loss = -(torch.stack(log_probs) * advantage).mean() - 0.01 * entropy
        value_loss = nn.MSELoss()(values_tensor, returns)
        
        policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        policy_optimizer.step()
        
        value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
        value_optimizer.step()
        
        rewards_per_episode.append(sum(episode_rewards))
        value_loss_history.append(value_loss.item())
        policy_loss_history.append(policy_loss.item())
        steps_per_episode.append(total_steps)
        
        if len(rewards_per_episode) >= 50 and np.mean(rewards_per_episode[-50:]) > -100:
            print(f"Solved after {episode} episodes!")
            break
    
    return rewards_per_episode, value_loss_history, policy_loss_history, steps_per_episode


def hyper_param_tuning_acrobot(env_name, device):
    results_dir = "acrobot_rf_results"
    os.makedirs(results_dir, exist_ok=True)
    
    hyperparameter_grid = {
        "num_episodes": [3000],
        "alpha_policy": [1e-2, 1e-3, 1e-4],
        "alpha_value": [0.1, 1e-3, 1e-2],
        "gamma": [0.99],
        "reward_scale": [1.0]  
    }
    hyperparameter_combinations = list(product(*hyperparameter_grid.values()))
    results = []
    
    for params in hyperparameter_combinations:
        hyper_params = dict(zip(hyperparameter_grid.keys(), params))
        folder_name = "_".join([f"{key}={value}" for key, value in hyper_params.items()])
        results_folder = os.path.join(results_dir, folder_name)
        os.makedirs(results_folder, exist_ok=True)

        rewards_across_seeds = []
        for seed in range(3): 
            rewards, _, _, _ = reinforce_with_baseline(
                gym.make(env_name),
                PolicyNetwork(6, 3).to(device),
                ValueNetwork(6).to(device),
                hyper_params['num_episodes'],
                hyper_params['gamma'],
                hyper_params['alpha_policy'],
                hyper_params['alpha_value'],
                seed=seed,
                reward_scale=hyper_params['reward_scale']
            )
            rewards_across_seeds.append(np.mean(rewards[-50:]))
        
        mean_reward_across_seeds = np.mean(rewards_across_seeds)
        results.append((hyper_params, mean_reward_across_seeds, results_folder))
    
    
    best_params, _, best_folder = max(results, key=lambda x: x[1])
    print(f"Best Hyperparameters: {best_params}")
    return best_params

# Function Definition
def evaluate_with_multiple_seeds(env_name, device, best_hyperparameters, save_gif=False, gif_filename="policy_visualization.gif"):
    successful_runs = 0
    total_runs = 5
    frames = []

    for seed in range(total_runs):
        env = gym.make(env_name, render_mode="rgb_array")
        policy_net = PolicyNetwork(6, 3).to(device)
        value_net = ValueNetwork(6).to(device)

        rewards, _, _, _ = reinforce_with_baseline(
            env,
            policy_net,
            value_net,
            best_hyperparameters['num_episodes'],
            best_hyperparameters['gamma'],
            best_hyperparameters['alpha_policy'],
            best_hyperparameters['alpha_value'],
            seed=seed,
            reward_scale=best_hyperparameters.get('reward_scale', 1.0)
        )

        if np.mean(rewards[-50:]) > -100:
            successful_runs += 1

        state, _ = env.reset()
        total_reward = 0
        while True:
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action_probs = policy_net(state_tensor)
                action = torch.argmax(action_probs).item()

            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            if save_gif:
                frame = env.render()
                frames.append(frame)

            state = next_state
            if done or truncated:
                print(f"Run {seed + 1} completed. Total reward: {total_reward}")
                break

        env.close()

    if save_gif and frames:
        print(f"Saving GIF as {gif_filename}...")
        imageio.mimsave(gif_filename, frames, fps=30)

    print(f"Success rate: {successful_runs}/{total_runs}")
    return successful_runs



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_name = 'Acrobot-v1'

best_hyperparameters = hyper_param_tuning_acrobot(env_name, device)

success_rate = evaluate_with_multiple_seeds(
    env_name,
    device,
    best_hyperparameters,
    save_gif=True, 
    gif_filename="acrobot_rf_results/acrobot_rf.gif"
)