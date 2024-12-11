import torch
import gym
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import warnings
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)
import time

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(F"Device: {device}")
class PolicyParameterNet(nn.Module):
    def __init__(self,stateSize,actionSize,hiddenLayerSize):
        super(PolicyParameterNet,self).__init__()
        self.fc1 = nn.Linear(stateSize,hiddenLayerSize)
        self.fc2 = nn.Linear(hiddenLayerSize,actionSize)
        self.to(device)
    
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x),dim=0)
        return x

class ValueParameterNet(nn.Module):
    def __init__(self,stateSize,hiddenLayerSize,outputSize):
        super(ValueParameterNet,self).__init__()
        self.fc1 = nn.Linear(stateSize,hiddenLayerSize)
        self.fc2 = nn.Linear(hiddenLayerSize,1)
        self.to(device)
    
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ActorCritic:
    def __init__(self,envName,hiddenLayerSize,episodes=2000,policyStepSize=0.0001,valueStepSize=0.0001,runs=10,gamma=0.99):
        self.envName = envName
        self.env = gym.make(envName)
        self.states,info = self.env.reset()
         
        self.episodes = episodes
        self.pAlpha = policyStepSize # Step size for policy parameter update
        self.vAlpha = valueStepSize # Step size for value parameter update
        self.gamma = gamma
        self.runs = runs
        self.hiddenLayerSize = hiddenLayerSize

    def train(self):
        stateSize = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            actionSize = self.env.action_space.n
        elif isinstance(self.env.action_space, gym.spaces.Box):
            actionSize = self.env.action_space.shape[0]  # Use number of dimensions
        else:
            raise NotImplementedError("Unknown action space type")

        avgRewards = torch.zeros((self.runs,self.episodes))
        totalActionsPerRun = torch.zeros((self.runs,self.episodes))
        for run in range(self.runs):
            policyNetwork = PolicyParameterNet(stateSize,actionSize,self.hiddenLayerSize)
            valueNetwork = ValueParameterNet(stateSize,self.hiddenLayerSize,1)

            policyOptimizer = optim.Adam(policyNetwork.parameters(),lr=self.pAlpha)
            valueOptimizer = optim.Adam(valueNetwork.parameters(),lr=self.vAlpha)

            episodeRewards = []
            totalActionsPerEpisode = []
            actions = 0
            for episode in range(self.episodes):
                initialState,info = self.env.reset()
                state = torch.tensor(initialState,dtype=torch.float32).to(device)
                rewards = []
                factor = 1
                
                while True:
                    actionProbs = policyNetwork(state)
                    action = np.random.choice(actionSize, p=actionProbs.detach().cpu().numpy())
                    
                    nextState, reward, terminated, truncated, _ = self.env.step(action)
                    
                    rewards.append(reward)

                    valueFunction = valueNetwork(state)
                    valueFunctionNext = valueNetwork(torch.tensor(nextState,dtype=torch.float32).to(device))

                    if terminated or truncated:
                        valueFunctionNext = torch.tensor([0],dtype=torch.float32).unsqueeze(0).to(device)
                    
                    tdError = reward + self.gamma*(valueFunctionNext) - valueFunction

                    actions += 1
                    # Updating Value Parameters
                    valueOptimizer.zero_grad()
                    valueLoss = -tdError.item()*valueFunction
                    valueLoss.backward()
                    valueOptimizer.step()

                    # Updating Policy Parameters
                    policyOptimizer.zero_grad()
                    policyLoss = -torch.log(actionProbs[action])*tdError.item()*factor
                    policyLoss.backward(retain_graph=True)
                    policyOptimizer.step()

                    state = torch.tensor(nextState,dtype=torch.float32).to(device)
                    factor = self.gamma*factor
                    if terminated or truncated:
                        break
                
                episodeReward = torch.sum(torch.tensor(rewards))
                episodeRewards.append(episodeReward)
                totalActionsPerEpisode.append(actions)
                if episode%100 == 0:
                    print(f'Run: {run}, Episode: {episode}, Reward: {episodeReward}')
                self.env.close()
            totalActionsPerRun[run] = torch.tensor(totalActionsPerEpisode)
            avgRewards[run] = torch.tensor(episodeRewards)
        return avgRewards, totalActionsPerRun
    
    def build(self):
        rewards,actions = self.train()
        return rewards, actions

    def plotRewardsWithStd(self, rewards):
   
        rewards = rewards.numpy()
        
        std = np.std(rewards, axis=0)  
        avgRewards = np.mean(rewards, axis=0)
        
        plt.plot(avgRewards,label='Mean Return')
        plt.fill_between(range(self.episodes), avgRewards-std, avgRewards+std, color='blue', alpha=0.2, label='Std Dev')
        #plt.errorbar(range(self.episodes), avgRewards, yerr=std, fmt='-o', ecolor='b', capsize=5,label='Mean Return ± Std Dev')
        plt.xlabel('Episodes')
        plt.ylabel('Average Rewards')
        plt.title('Average Rewards for Actor Critic with Standard Deviation')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'ActorCritic_Std_gamma:{self.gamma}_{self.envName[:-3]}_Adam_{self.pAlpha}_{self.vAlpha}_hlAyers{self.hiddenLayerSize}.png')
        plt.close()
    
    def plotRewards(self, rewards):
   
        rewards = rewards.numpy()
        
        avgRewards = np.mean(rewards, axis=0)
        
        plt.plot(avgRewards)
        #plt.errorbar(range(self.episodes), avgRewards, yerr=std, fmt='-o', ecolor='r', capsize=5,label='Mean Return ± Std Dev')
        plt.xlabel('Episodes')
        plt.ylabel('Average Rewards')
        plt.title('Actor Critic')
        plt.grid(True)
        plt.savefig(f'ActorCritic_{self.envName}_Adam_{self.pAlpha}_{self.vAlpha}.png')
        plt.close()

    def plotActions(self, actions):
        actions = actions.numpy()
        avgActions = np.mean(actions, axis=0)
        plt.plot(avgActions)
        plt.xlabel('Episodes')
        plt.ylabel('Average Actions')
        plt.title(f'Total Actions for {self.envName[:-3]} using Actor Critic')
        plt.grid(True)
        plt.savefig(f'ActorCritic_Actions_gamma:{self.gamma}_{self.envName[:-3]}_Adam_{self.pAlpha}_{self.vAlpha}_hlAyers{self.hiddenLayerSize}.png')
        plt.close()

if __name__ == '__main__':
    acrobat = 'Acrobot-v1'
    cartPole = 'CartPole-v1'
    mountainCar = 'MountainCar-v0'

    

    gammas = [0.99,0.5,0.1]
    hiddenLayerSizes = [32,64,128]
    policyStepSizes = [0.0001,0.001,0.01]
    valueStepSizes = [0.0001,0.001,0.01]
    runs = 10
    episodes = 1000

    
    #Fixing Gamma , policyStepSize, valueStepSize and varying hiddenLayerSize
    # for hiddenLayerSize in hiddenLayerSizes:
    #     time1 = time.time()
    #     gamma = 0.99
    #     policyStepSize = 0.0001
    #     valueStepSize = 0.001
    #     print(f'Gamma: {gamma}, HiddenLayerSize: {hiddenLayerSize}, PolicyStepSize: {policyStepSize}, ValueStepSize: {valueStepSize}')
    #     actorCritic = ActorCritic(acrobat,hiddenLayerSize=hiddenLayerSize,policyStepSize=policyStepSize,valueStepSize=valueStepSize,episodes=episodes,runs=runs)
    #     rewards,actions = actorCritic.build()
    #     actorCritic.plotRewardsWithStd(rewards)
    #     actorCritic.plotActions(actions)
    #     time2 = time.time()
    #     print(f'Time taken: {time2-time1}')
    #     with open('AcrobotExperiments.txt','a') as file:
    #         file.write(f"Gamma: {gamma}, HiddenLayerSize: {hiddenLayerSize}, PolicyStepSize: {policyStepSize}, ValueStepSize: {valueStepSize}, Time: {time2-time1}\n")
        
    #Fixing HiddenLayerSize, policyStepSize, valueStepSize and varying gamma
    #Fixing Gamma , policyStepSize, valueStepSize and varying hiddenLayerSize
    for gamma in gammas:
        time1 = time.time()
        hiddenLayerSize = 32
        policyStepSize = 0.0001
        valueStepSize = 0.001
        runs = 5
        print(f'Gamma: {gamma}, HiddenLayerSize: {hiddenLayerSize}, PolicyStepSize: {policyStepSize}, ValueStepSize: {valueStepSize}')
        actorCritic = ActorCritic(acrobat,hiddenLayerSize=hiddenLayerSize,policyStepSize=policyStepSize,valueStepSize=valueStepSize,episodes=episodes,runs=runs,gamma=gamma)
        rewards,actions = actorCritic.build()
        actorCritic.plotRewardsWithStd(rewards)
        actorCritic.plotActions(actions)
        time2 = time.time()
        print(f'Time taken: {time2-time1}')
        with open('AcrobotExperiments.txt','a') as file:
            file.write(f"Gamma: {gamma}, HiddenLayerSize: {hiddenLayerSize}, PolicyStepSize: {policyStepSize}, ValueStepSize: {valueStepSize}, Time: {time2-time1}\n")
      
    # #Fixing HiddenLayerSize, gamma, valueStepSize and varying policyStepSize
    # for policyStepSize in policyStepSizes:
    #     time1 = time.time()
    #     hiddenLayerSize = 32
    #     gamma = 0.99
    #     valueStepSize = 0.001
    #     print(f'Gamma: {gamma}, HiddenLayerSize: {hiddenLayerSize}, PolicyStepSize: {policyStepSize}, ValueStepSize: {valueStepSize}')
    #     actorCritic = ActorCritic(acrobat,hiddenLayerSize=hiddenLayerSize,policyStepSize=policyStepSize,valueStepSize=valueStepSize,episodes=episodes,runs=runs,gamma=gamma)
    #     rewards,actions = actorCritic.build()
    #     actorCritic.plotRewardsWithStd(rewards)
    #     actorCritic.plotActions(actions)
    #     time2 = time.time()
    #     print(f'Time taken: {time2-time1}')
    #     with open('AcrobotExperiments.txt','a') as file:
    #         file.write(f"Gamma: {gamma}, HiddenLayerSize: {hiddenLayerSize}, PolicyStepSize: {policyStepSize}, ValueStepSize: {valueStepSize}, Time: {time2-time1}\n")

    # #Fixing HiddenLayerSize, gamma, policyStepSize and varying valueStepSize
    # for valueStepSize in valueStepSizes:
    #     time1 = time.time()
    #     hiddenLayerSize = 32
    #     gamma = 0.99
    #     policyStepSize = 0.0001
    #     print(f'Gamma: {gamma}, HiddenLayerSize: {hiddenLayerSize}, PolicyStepSize: {policyStepSize}, ValueStepSize: {valueStepSize}')
    #     actorCritic = ActorCritic(acrobat,hiddenLayerSize=hiddenLayerSize,policyStepSize=policyStepSize,valueStepSize=valueStepSize,episodes=episodes,runs=runs,gamma=gamma)
    #     rewards,actions = actorCritic.build()
    #     actorCritic.plotRewardsWithStd(rewards)
    #     actorCritic.plotActions(actions)
    #     time2 = time.time()
    #     print(f'Time taken: {time2-time1}')
    #     with open('AcrobotExperiments.txt','a') as file:
    #         file.write(f"Gamma: {gamma}, HiddenLayerSize: {hiddenLayerSize}, PolicyStepSize: {policyStepSize}, ValueStepSize: {valueStepSize}, Time: {time2-time1}\n")

        
    #running with best observed hyperparameters
    # time1 = time.time()
    # hiddenLayerSize = 128   
    # gamma = 0.99
    # policyStepSize = 0.0001
    # valueStepSize = 0.01
    # print(f'Gamma: {gamma}, HiddenLayerSize: {hiddenLayerSize}, PolicyStepSize: {policyStepSize}, ValueStepSize: {valueStepSize}')
    # actorCritic = ActorCritic('CartPole-v1',hiddenLayerSize=hiddenLayerSize,policyStepSize=policyStepSize,valueStepSize=valueStepSize,episodes=episodes,runs=runs,gamma=gamma)
    # rewards,actions = actorCritic.build()
    # actorCritic.plotRewardsWithStd(rewards)
    # actorCritic.plotActions(actions)
    # time2 = time.time()
    # print(f'Time taken: {time2-time1}')
    # with open('CartPoleExperiments.txt','a') as file:
    #     file.write(f"Gamma: {gamma}, HiddenLayerSize: {hiddenLayerSize}, PolicyStepSize: {policyStepSize}, ValueStepSize: {valueStepSize}, Time: {time2-time1}\n")
    

    #running with best observed hyperparameters
    # time1 = time.time()
    # hiddenLayerSize = 128   
    # gamma = 0.99
    # policyStepSize = 0.0001
    # valueStepSize = 0.001
    # print(f'Gamma: {gamma}, HiddenLayerSize: {hiddenLayerSize}, PolicyStepSize: {policyStepSize}, ValueStepSize: {valueStepSize}')
    # actorCritic = ActorCritic('CartPole-v1',hiddenLayerSize=hiddenLayerSize,policyStepSize=policyStepSize,valueStepSize=valueStepSize,episodes=episodes,runs=runs,gamma=gamma)
    # rewards,actions = actorCritic.build()
    # actorCritic.plotRewardsWithStd(rewards)
    # actorCritic.plotActions(actions)
    # time2 = time.time()
    # print(f'Time taken: {time2-time1}')
    # with open('CartPoleExperiments.txt','a') as file:
    #     file.write(f"Gamma: {gamma}, HiddenLayerSize: {hiddenLayerSize}, PolicyStepSize: {policyStepSize}, ValueStepSize: {valueStepSize}, Time: {time2-time1}\n")
    