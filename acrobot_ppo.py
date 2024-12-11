import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import deque
import gym
import torch.nn.functional as F
from torch.distributions import Categorical
import imageio

HYPERPARAMETERS = {
    'Acrobot-v1' : {
        'horizon': 501,
        'total_timesteps': 100000,
        'no_of_batch_timesteps': 2000,
        'no_of_batch_episodes': 10,
        'no_of_episodes': 500,
        'gamma' : 0.99,
        'solved_score' : -100,
        'policy_learning_rate' : 1e-3,
        'value_learning_rate' : 5e-3,
        'no_of_epochs': 10,
        'epsilon' : 0.2,
        'batch_size': 256,
    }
}

def initialize_random_seeds(seed=42):
    """ Set random seeds for reproducibility. """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class RolloutBuffer(Dataset):
    """ Dataset to store rollouts for PPO training. """
    def __init__(self, observations, actions, log_probs, rewards):
        self.observations = observations
        self.actions = actions
        self.log_probs = log_probs
        self.rewards = rewards

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (self.observations[idx], 
                self.actions[idx], 
                self.log_probs[idx], 
                self.rewards[idx])

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

class PPO:
    def __init__(self, env, rand_seed=42, root_dir='acrobot_ppo_logs', eval_mode=False):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.hyper_params = HYPERPARAMETERS[env.spec.id]
        self.eval_mode=eval_mode
        
        self.policy_network = PolicyNetwork(self.state_dim, self.action_dim)
        self.value_network = ValueNetwork(self.state_dim)
        
        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=self.hyper_params['policy_learning_rate'])
        self.optimizer_value = optim.Adam(self.value_network.parameters(), lr=self.hyper_params['value_learning_rate'])
        
        initialize_random_seeds(rand_seed)
        
        self.recent_episode_rewards = deque(maxlen=10)
        self.no_episodes = 0
        self.ep_rewards = []
        self.time_steps = 0
        
        self.model_dir = os.path.join(root_dir, env.spec.id)
        self.best_mean_reward = -float('inf')
        os.makedirs(self.model_dir, exist_ok=True)
        
    def save_models(self):
        torch.save(self.policy_network.state_dict(), os.path.join(self.model_dir, 'policy_network.pth'))
        torch.save(self.value_network.state_dict(), os.path.join(self.model_dir, 'value_network.pth'))

    def log_statistics(self):
        """ Log training statistics. """
        recent_avg_reward = np.mean(self.recent_episode_rewards) if len(self.recent_episode_rewards) > 0 else 0
        print(f"Episode: {self.no_episodes}, Recent Average Reward: {recent_avg_reward:.2f}, Best Average Reward: {self.best_mean_reward:.2f}")

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_network(state_tensor)

        if not self.eval_mode:
            dist = Categorical(action_probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item()
        
        return action_probs.argmax().item(), None

    def get_discounted_rewards(self, batch_rewards):
        discounted_rewards = []
        
        for episode_rewards in batch_rewards[::-1]:
            cumulative_reward = 0
            for reward in episode_rewards[::-1]:
                cumulative_reward = reward + self.hyper_params['gamma'] * cumulative_reward
                discounted_rewards.append(cumulative_reward)

        return discounted_rewards[::-1]  # Reverse to maintain order

    def collect_rollouts(self):
        states, actions, log_action_probs = [], [], []
        batch_rewards = []

        while len(batch_rewards) < self.hyper_params['no_of_batch_episodes']:
            state, _ = self.env.reset()
            ep_reward = 0
            ep_rewards_arr = []

            for _ in range(self.hyper_params['horizon']):
                action, log_prob = self.get_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)

                # Reward Shaping
                shaped_reward = reward + 0.1 * (abs(next_state[1]) - abs(state[1]))  # Example: Encourage angular velocity toward goal
                if done and reward == 0:  # Success case in Acrobot
                    shaped_reward += 10  # Add a completion bonus

                ep_reward += shaped_reward
                states.append(state)
                actions.append(action)
                log_action_probs.append(log_prob)
                ep_rewards_arr.append(shaped_reward)

                state = next_state

                if done or truncated:
                    break

            ep_rewards_arr = (np.array(ep_rewards_arr) - np.mean(ep_rewards_arr)) / (np.std(ep_rewards_arr) + 1e-8)

            batch_rewards.append(ep_rewards_arr.tolist())
            self.no_episodes += 1
            self.ep_rewards.append(ep_reward)
            self.recent_episode_rewards.append(ep_reward)

            self.log_statistics()

            if len(self.recent_episode_rewards) == 10:
                recent_avg_reward = np.mean(self.recent_episode_rewards)
                if recent_avg_reward > self.best_mean_reward:
                    self.best_mean_reward = recent_avg_reward
                    self.save_models()

        return states, actions, log_action_probs, batch_rewards


    def get_action_probs_and_state_values(self, states_tensor, actions_tensor):
        action_probs = self.policy_network(states_tensor)
        dist = Categorical(action_probs)

        actions_tensor = actions_tensor.unsqueeze(1)  
        action_probs_selected = action_probs.gather(1, actions_tensor).squeeze()
        state_values = self.value_network(states_tensor).squeeze()

        return torch.log(action_probs_selected), state_values, dist.entropy()

    def train(self):
        while self.time_steps < self.hyper_params['total_timesteps'] and self.no_episodes < self.hyper_params['no_of_episodes']:
            states, actions, log_action_probs, batch_rewards = self.collect_rollouts()
            discounted_rewards = self.get_discounted_rewards(batch_rewards)
            old_action_log_probs = torch.tensor(log_action_probs, dtype=torch.float32)

            rollout_dataset = RolloutBuffer(states, actions, old_action_log_probs.numpy(), discounted_rewards)
            rollout_dataloader = DataLoader(rollout_dataset, batch_size=self.hyper_params['batch_size'], shuffle=True)

            for epoch in range(self.hyper_params['no_of_epochs']):
                for batch in rollout_dataloader:
                    batch_states, batch_actions, batch_old_action_log_probs, batch_target_rewards = batch

                    new_action_log_probs, state_values, _ = self.get_action_probs_and_state_values(batch_states.float(), batch_actions)

                    ratios = torch.exp(new_action_log_probs - batch_old_action_log_probs)
                    advantages = batch_target_rewards - state_values.detach()

                    if advantages.numel() <= 1:
                        continue

                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

                    surrogate_loss_1 = ratios * advantages
                    surrogate_loss_2 = torch.clamp(ratios,
                                                    1 - self.hyper_params['epsilon'],
                                                    1 + self.hyper_params['epsilon']) * advantages

                    policy_loss = -torch.min(surrogate_loss_1.mean(), surrogate_loss_2.mean())
                    value_loss = F.mse_loss(batch_target_rewards.float(), state_values.float())

                    # Update policy network
                    self.optimizer_policy.zero_grad()
                    policy_loss.backward()
                    self.optimizer_policy.step()

                    # Update value network
                    self.optimizer_value.zero_grad()
                    value_loss.backward()
                    self.optimizer_value.step()


    def eval(self, no_episodes=10, gif_path="acrobot_ppo_logs/evaluation_visualization.gif"):
        """Evaluate the PPO agent and save visualization as a GIF."""
        # Load the trained policy and value networks
        self.policy_network.load_state_dict(torch.load(os.path.join(self.model_dir, 'policy_network.pth')))
        self.value_network.load_state_dict(torch.load(os.path.join(self.model_dir, 'value_network.pth')))
        self.policy_network.eval()
        self.value_network.eval()

        total_rewards = []  # To store rewards for all evaluation episodes
        frames = []  # To store frames for the GIF

        with torch.no_grad():
            for ep in range(no_episodes):
                state, _ = self.env.reset()
                ep_reward = 0  # Initialize reward for this episode
                step_count = 0  # Count steps in the episode

                print(f"Starting episode {ep + 1}...")  # Debug: Start of episode

                while True:
                    # Capture frame for the GIF
                    frame = self.env.render()
                    frames.append(frame)

                    action, _ = self.get_action(state)  # Get action from policy
                    next_state, reward, done, truncated, _ = self.env.step(action)
                    ep_reward += reward  # Accumulate reward
                    step_count += 1
                    state = next_state
                    if done or truncated:  # Check if the episode is over
                        break

                total_rewards.append(ep_reward)
                print(f"Episode {ep + 1} finished with total reward: {ep_reward:.2f}")

        # Save the GIF
        print(f"Saving evaluation visualization as GIF to {gif_path}...")
        imageio.mimsave(gif_path, frames, fps=30)  # Adjust FPS as needed

        # Calculate and print the average reward over all episodes
        avg_reward = np.mean(total_rewards)
        print(f"\nAverage Reward over {no_episodes} episodes: {avg_reward:.2f}")
        return avg_reward


if __name__ == '__main__':
    env_name = 'Acrobot-v1'
    rand_seed = 0

    print(f"Training environment {env_name} with random seed {rand_seed}...")
    env = gym.make(env_name)
    agent = PPO(env=env, rand_seed=rand_seed)
    agent.train()
    env.close()

    print(f'Evaluating environment {env_name} with random seed {rand_seed}.... \n\n')
    env = gym.make(env_name, render_mode='rgb_array')  # Use 'rgb_array' for GIF rendering
    agent = PPO(env=env, rand_seed=rand_seed, eval_mode=True)
    gif_path = "acrobot_ppo_logs/ppo_acrobot_evaluation.gif"  # Path to save the GIF
    agent.eval(gif_path=gif_path)