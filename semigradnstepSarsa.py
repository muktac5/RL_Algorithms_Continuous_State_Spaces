import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import gym
import time

class QNetwork(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenUnits):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(inputSize, hiddenUnits)
        self.fc2 = nn.Linear(hiddenUnits, outputSize)
    
    def forward(self, state,action):
        x = torch.cat([state,action],dim=0)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SemiGradNStepSarsa():
    def __init__(self, env,states, actions, hiddenUnits=128, episodes=1000, gamma=0.99, alpha=0.001,runs=10,epsilon=0.1,n=5):
        self.envName = env
        self.env = gym.make(env)
        self.states = self.env.observation_space.shape[0]
        self.actions = self.env.action_space.n
        self.hiddenUnits = hiddenUnits
        self.episodes = episodes
        self.gamma = gamma
        self.alpha = alpha
        self.runs = runs
        self.epsilon = epsilon
        self.n = n
        self.qnetwork = QNetwork(states + 1, 1 , hiddenUnits)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=alpha)
    
    def getAction(self, state):
        QValues = []
        for action in range(self.actions):
            qhat = self.qnetwork(torch.FloatTensor(state),torch.FloatTensor([action])).detach().numpy()
            QValues.append(qhat)
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(QValues)
        


    def train(self):
        allReturns = np.zeros((self.runs, self.episodes))
        totalActionsPerRun = np.zeros((self.runs, self.episodes))
        for run in range(self.runs):
            returns = []
            totalActionsPerEpisode = []
            actionsCount = 0
            for episode in range(self.episodes):
                state,info = self.env.reset()
                action = self.getAction(state)
                states = [state]
                actions = [action]
                actionsCount += 1
                rewards = [0]
                timeStep = 0
                T = np.inf
                while True:
                    if timeStep < T:
                        nextState, reward, terminated, truncated, *_ = self.env.step(actions[timeStep])
                        states.append(nextState)
                        rewards.append(reward)
                        if terminated or truncated:
                            T = timeStep + 1
                        else:
                            nextAction = self.getAction(nextState)
                            actions.append(nextAction)
                            actionsCount += 1
                    tau = timeStep - self.n + 1

                    if tau >= 0:
                        G = 0
                        for i in range(tau + 1, min(tau + self.n, T)+1):
                            G += np.power(self.gamma, i-tau-1)*rewards[i]
                        
                        if tau + self.n < T:
                            G += np.power(self.gamma, self.n)*self.qnetwork(torch.FloatTensor(states[tau+self.n]),torch.FloatTensor([actions[tau+self.n]])).unsqueeze(0)
                        
                        else:
                            G = torch.tensor(G).float().unsqueeze(0)

                        qhat = self.qnetwork(torch.FloatTensor(states[tau]),torch.FloatTensor([actions[tau]])).unsqueeze(0)
                        delta = G - qhat
                        self.optimizer.zero_grad()
                        valueLoss = -delta.item()*qhat
                        valueLoss.backward()
                        self.optimizer.step()
                    timeStep += 1

                    if tau >= T - 1:
                        break                                                    
                
                rewardSum = np.sum(rewards)
                returns.append(rewardSum)
                totalActionsPerEpisode.append(actionsCount)
                if episode % 100 == 0:
                    print(f"Run: {run}, Episode: {episode}, Reward: {rewardSum}")
            allReturns[run] = returns
            totalActionsPerRun[run] = totalActionsPerEpisode
        return allReturns, totalActionsPerRun
    
    # def plotRewardsWithStd(self, rewards):
   
    #     #rewards = rewards.numpy()
        
    #     std = np.std(rewards, axis=0)  
    #     avgRewards = np.mean(rewards, axis=0)
        
    #     plt.plot(avgRewards,label='Mean Return')
    #     plt.fill_between(range(self.episodes), avgRewards-std, avgRewards+std, color='blue', alpha=0.5, label='Std Dev')
    #     #plt.errorbar(range(self.episodes), avgRewards, yerr=std, fmt='-o', ecolor='b', capsize=5,label='Mean Return ± Std Dev')
    #     plt.xlabel('Episodes')
    #     plt.ylabel('Average Rewards')
    #     plt.title('Average Rewards for Semi Gradient-NStep Sarsa with Standard Deviation')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(f'SGNS_Std_gamma:{self.gamma}_{self.envName[:-3]}_Adam_{self.alpha}_hlAyers{self.hiddenUnits}.png')
    #     plt.close()
    def plotRewardsWithStd(self,rewards):
        std = np.std(rewards, axis=0)
        avgRewards = np.mean(rewards, axis=0)
        
       
        # Plot with improved colors and smoother lines
        plt.figure(figsize=(10, 6))  # Increase figure size
        plt.plot(avgRewards, label='Mean Return', linewidth=2)
        plt.fill_between(
            range(self.episodes), 
            avgRewards - std, 
            avgRewards + std, 
            color='lightskyblue', 
            alpha=0.4, 
            label='Std Dev'
        )
        plt.xlabel('Episodes', fontsize=12)
        plt.ylabel('Average Rewards', fontsize=12)
        plt.title('Average Rewards for Semi Gradient-NStep Sarsa with Standard Deviation', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.legend()
        plt.savefig(f'SGNS_Std_gamma:{self.gamma}_{self.envName[:-3]}_Adam_{self.alpha}_hlAyers{self.hiddenUnits}_epsilon{self.epsilon}_n{self.n}_Cartpole.png')
        plt.close()

    def plotActions(self, actions):
        #actions = actions.numpy()
        avgActions = np.mean(actions, axis=0)
        plt.plot(avgActions)
        plt.xlabel('Episodes')
        plt.ylabel('Average Actions')
        plt.title(f'Total Actions for {self.envName[:-3]} using Semi Gradient-NStep Sarsa')
        plt.grid(True)
        plt.savefig(f'SGNS_Actions_gamma:{self.gamma}_{self.envName[:-3]}_Adam_{self.alpha}_hlAyers{self.hiddenUnits}_epsilon{self.epsilon}_n{self.n}_Cartpole.png')
        plt.close()
    

    def build(self):
        allReturns,allActions = self.train()
        self.plotRewardsWithStd(allReturns)
        self.plotActions(allActions)

        return

if __name__ == '__main__':
    #env = 'CartPole-v1'
    # states = 4
    # actions = 2
    env = 'Acrobot-v1'
    states = 6
    actions = 3
    hiddenUnits = [32,64,128]
    episodes = 1000
    gamma = 0.99
    alpha = 0.001
    runs = 5
    epsilon = 0.1
    n = 5
    hiddenUnits = 256
    semigradnstepSarsa = SemiGradNStepSarsa(env,states, actions, hiddenUnits, episodes, gamma, alpha, runs, epsilon, n)
    semigradnstepSarsa.build()

    #Fixing gamma and alpha, n, varying hidden units
    # hiddenUnits = [64,128,256]
    # for units in hiddenUnits:
    #     time1 = time.time()
    #     gamma = 0.99
    #     alpha = 0.001
    #     epsilon = 0.1
    #     n = 5
    #     print(f"GAMMA: {gamma}, ALPHA: {alpha}, EPSILON: {epsilon}, N: {n}, HIDDEN UNITS: {units}")
    #     semigradnstepSarsa = SemiGradNStepSarsa(env,states, actions, units, episodes, gamma, alpha, runs, epsilon, n)
    #     semigradnstepSarsa.build()
    #     time2 = time.time()
    #     print(f'Time taken for hidden units {units} is {time2-time1}')
    #     with open("SGNS.txt", "a") as f:
    #         f.write(f"Gamma: {gamma}, Alpha: {alpha}, Epsilon: {epsilon}, N: {n}, Hidden Units: {units}, Time: {time2-time1}\n")

    #Fixing hiddenunits and alpha, n, varying gamma
    # gammas = [0.99,0.8,0.4]
    # for gamma in gammas:
    #     time1 = time.time()
    #     units = 128
    #     alpha = 0.001
    #     epsilon = 0.1
    #     n = 5
    #     print(f"GAMMA: {gamma}, ALPHA: {alpha}, EPSILON: {epsilon}, N: {n}, HIDDEN UNITS: {units}")
    #     semigradnstepSarsa = SemiGradNStepSarsa(env,states, actions, units, episodes, gamma, alpha, runs, epsilon, n)
    #     semigradnstepSarsa.build()
    #     time2 = time.time()
    #     print(f'Time taken for hidden units {units} is {time2-time1}')
    #     with open("SGNS.txt", "a") as f:
    #         f.write(f"Gamma: {gamma}, Alpha: {alpha}, Epsilon: {epsilon}, N: {n}, Hidden Units: {units}, Time: {time2-time1}\n")

    #Fixing hiddenunits and gamma, n, varying alpha
    # alphas = [0.1,0.01,0.001]
    # for alpha in alphas:
    #     time1 = time.time()
    #     units = 128
    #     gamma = 0.99
    #     epsilon = 0.1
    #     n = 5
    #     print(f"GAMMA: {gamma}, ALPHA: {alpha}, EPSILON: {epsilon}, N: {n}, HIDDEN UNITS: {units}")
    #     semigradnstepSarsa = SemiGradNStepSarsa(env,states, actions, units, episodes, gamma, alpha, runs, epsilon, n)
    #     semigradnstepSarsa.build()
    #     time2 = time.time()
    #     print(f'Time taken for hidden units {units} is {time2-time1}')
    #     with open("SGNS.txt", "a") as f:
    #         f.write(f"Gamma: {gamma}, Alpha: {alpha}, Epsilon: {epsilon}, N: {n}, Hidden Units: {units}, Time: {time2-time1}\n")

    #Fixing hiddenunits and gamma, alpha, varying epsilon
    # epsilons = [0.1,0.5,0.9]
    # for epsilon in epsilons:
    #     time1 = time.time()
    #     units = 128
    #     gamma = 0.99
    #     alpha = 0.001
    #     n = 5
    #     print(f"GAMMA: {gamma}, ALPHA: {alpha}, EPSILON: {epsilon}, N: {n}, HIDDEN UNITS: {units}")
    #     semigradnstepSarsa = SemiGradNStepSarsa(env,states, actions, units, episodes, gamma, alpha, runs, epsilon, n)
    #     semigradnstepSarsa.build()
    #     time2 = time.time()
    #     print(f'Time taken for hidden units {units} is {time2-time1}')
    #     with open("SGNS.txt", "a") as f:
    #         f.write(f"Gamma: {gamma}, Alpha: {alpha}, Epsilon: {epsilon}, N: {n}, Hidden Units: {units}, Time: {time2-time1}\n")

    #Fixing hiddenunits and gamma, alpha, epsilon, varying n
    # ns = [5,10,15]
    # for n in ns:
    #     time1 = time.time()
    #     units = 128
    #     gamma = 0.99
    #     alpha = 0.001
    #     epsilon = 0.1
    #     runs = 5
    #     print(f"GAMMA: {gamma}, ALPHA: {alpha}, EPSILON: {epsilon}, N: {n}, HIDDEN UNITS: {units}")
    #     semigradnstepSarsa = SemiGradNStepSarsa(env,states, actions, units, episodes, gamma, alpha, runs, epsilon, n)
    #     semigradnstepSarsa.build()
    #     time2 = time.time()
    #     print(f'Time taken for hidden units {units} is {time2-time1}')
    #     with open("SGNS.txt", "a") as f:
    #         f.write(f"Gamma: {gamma}, Alpha: {alpha}, Epsilon: {epsilon}, N: {n}, Hidden Units: {units}, Time: {time2-time1}\n")

    #best hyperparameters
    # time1 = time.time()
    # units = 128
    # gamma = 0.99
    # alpha = 0.001
    # epsilon = 0.1
    # n = 5
    # runs = 5
    # print(f"GAMMA: {gamma}, ALPHA: {alpha}, EPSILON: {epsilon}, N: {n}, HIDDEN UNITS: {units}")
    # semigradnstepSarsa = SemiGradNStepSarsa(env,states, actions, units, episodes, gamma, alpha, runs, epsilon, n)
    # semigradnstepSarsa.build()
    # time2 = time.time()
    # print(f'Time taken for hidden units {units} is {time2-time1}')
    # with open("SGNS.txt", "a") as f:
    #     f.write(f"Gamma: {gamma}, Alpha: {alpha}, Epsilon: {epsilon}, N: {n}, Hidden Units: {units}, Time: {time2-time1}\n")