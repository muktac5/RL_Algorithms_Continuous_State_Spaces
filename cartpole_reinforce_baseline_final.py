import os
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time
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


def reinforce_with_baseline(env, policy_net, value_net, num_episodes, gamma, alpha_policy, alpha_value, reward_scale=0.01):
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
        
        values_tensor = torch.stack(values).squeeze()
        advantage = returns - values_tensor.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        policy_loss = -(torch.stack(log_probs) * advantage).mean()
        value_loss = nn.MSELoss()(values_tensor, returns)
        
        policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        policy_optimizer.step()
        
        value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
        value_optimizer.step()
        
        rewards_per_episode.append(sum(episode_rewards) / reward_scale)
        value_loss_history.append(value_loss.item())
        policy_loss_history.append(policy_loss.item())
        steps_per_episode.append(total_steps)
        
        if len(rewards_per_episode) >= 10 and np.mean(rewards_per_episode[-10:]) > 495:
            print(f"Solved after {episode} episodes!")
            break
    
    return rewards_per_episode, value_loss_history, policy_loss_history, steps_per_episode


def save_plot(plot_func, filename, *args, **kwargs):
    plt.figure()
    plot_func(*args, **kwargs)
    plt.savefig(filename)
    plt.close()


def plot_cumulative_steps(steps_per_episode):
    cumulative_steps = np.cumsum(steps_per_episode)
    plt.plot(cumulative_steps, label="Cumulative Steps")
    plt.title("Cumulative Steps Over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.legend()


def plot_mean_and_std_rewards(rewards, window_size=10):
    means, std_devs = [], []
    for i in range(len(rewards)):
        window = rewards[max(0, i - window_size):(i + 1)]
        means.append(np.mean(window))
        std_devs.append(np.std(window))
    
    plt.plot(range(len(means)), means, label="Mean Reward")
    plt.fill_between(
        range(len(means)),
        np.array(means) - np.array(std_devs),
        np.array(means) + np.array(std_devs),
        alpha=0.2,
        label="Standard Deviation"
    )
    plt.title("Mean and Standard Deviation of Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()


def train_and_save(env_name, device, hyper_params, results_folder):
    env = gym.make(env_name)
    input_size, output_size = env.observation_space.shape[0], env.action_space.n
    
    policy_net = PolicyNetwork(input_size, output_size).to(device)
    value_net = ValueNetwork(input_size).to(device)
    
    rewards, value_losses, policy_losses, steps = reinforce_with_baseline(
        env, policy_net, value_net,
        num_episodes=hyper_params['num_episodes'],
        gamma=hyper_params['gamma'],
        alpha_policy=hyper_params['alpha_policy'],
        alpha_value=hyper_params['alpha_value'],
        reward_scale=hyper_params.get('reward_scale', 0.01)
    )
    
    # Create results folder and save the policy network
    os.makedirs(results_folder, exist_ok=True)
    torch.save(policy_net.state_dict(), os.path.join(results_folder, "policy_net.pth"))
    print(f"Model saved at {os.path.join(results_folder, 'policy_net.pth')}")
    
    # Save hyperparameters
    with open(os.path.join(results_folder, "hyperparams.txt"), "w") as f:
        for key, value in hyper_params.items():
            f.write(f"{key}: {value}\n")
    
    save_plot(plot_cumulative_steps, os.path.join(results_folder, "cumulative_steps.png"), steps)
    save_plot(plot_mean_and_std_rewards, os.path.join(results_folder, "mean_std_rewards.png"), rewards)
    
    return rewards



def hyper_param_tuning(env_name, device):
    results_dir = "cartpole_rf_results"
    os.makedirs(results_dir, exist_ok=True)
    
    hyperparameter_grid = {
        "num_episodes": [3000],
        "alpha_policy": [1e-2, 1e-3, 0.005],
        "alpha_value": [1e-3, 0.01, 0.005],
        "gamma": [0.99],
        "reward_scale": [0.01, 0.005]
    }
    hyperparameter_combinations = list(product(*hyperparameter_grid.values()))
    results = []
    
    for params in hyperparameter_combinations:
        hyper_params = dict(zip(hyperparameter_grid.keys(), params))
        folder_name = "_".join([f"{key}={value}" for key, value in hyper_params.items()])
        results_folder = os.path.join(results_dir, folder_name)
        print(f"Training with hyperparameters: {hyper_params}")
        rewards = train_and_save(env_name, device, hyper_params, results_folder)
        mean_reward = np.mean(rewards[-10:])
        results.append((hyper_params, mean_reward, results_folder))
    
    # Find and save the best model
    best_params, _, best_folder = max(results, key=lambda x: x[1])
    best_target_folder = os.path.join(results_dir, "best_hyperparameters")
    
    # Handle existing "best_hyperparameters" folder
    if os.path.exists(best_target_folder):
        import shutil
        shutil.rmtree(best_target_folder)
    
    os.rename(best_folder, best_target_folder)
    print(f"Best model saved to {best_target_folder}")
    return best_params


def evaluate_agent(env_name, policy_net_path, num_episodes, device, render_mode='rgb_array', gif_path='evaluation.gif'):
    if not os.path.exists(policy_net_path):
        raise FileNotFoundError(f"Model file not found at {policy_net_path}")
    
    env = gym.make(env_name, render_mode=render_mode)
    input_size, output_size = env.observation_space.shape[0], env.action_space.n

    policy_net = PolicyNetwork(input_size, output_size).to(device)
    policy_net.load_state_dict(torch.load(policy_net_path, map_location=device))
    policy_net.eval()

    rewards = []
    frames = [] 

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        print(f"Starting Episode {episode + 1}")
        while not done:
            # Capture frame for GIF
            frame = env.render()
            frames.append(frame)

            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action_probs = policy_net(state_tensor)
            action = torch.distributions.Categorical(action_probs).sample()

            next_state, reward, done, truncated, _ = env.step(action.item())
            episode_reward += reward

            state = next_state
            if done or truncated:
                break

        rewards.append(episode_reward)
        print(f"Episode {episode + 1} Reward: {episode_reward}")

    avg_reward = np.mean(rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")

    # Save frames as GIF
    print(f"Saving GIF to {gif_path}")
    imageio.mimsave(gif_path, frames, fps=30)  # Adjust FPS if necessary

    env.close()
    return rewards, avg_reward



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_name = 'CartPole-v1'
best_hyperparameters = hyper_param_tuning(env_name, device)
policy_net_path = "cartpole_rf_results/best_hyperparameters/policy_net.pth"
num_evaluation_episodes = 10

gif_path = "cartpole_rf_results/cartpole_evaluation.gif"
rewards, avg_reward = evaluate_agent(env_name, policy_net_path, num_evaluation_episodes, device, render_mode='rgb_array', gif_path=gif_path)

with open("evaluation_results.txt", "w") as f:
    f.write(f"Average Reward over {num_evaluation_episodes} episodes: {avg_reward}\n")
    f.write(f"Rewards: {rewards}\n")
