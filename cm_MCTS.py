import numpy as np
import random
from collections import defaultdict

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.visits = 0
        self.wins = 0
        self.children = {}

class CatVsMonsters:
    def __init__(self):
        self.rows, self.cols = 5, 5
        
        # Define optimal policy first
        self.optimalPolicy = np.array([['AR','AD','AL','AD','AD'],
                                       ['AR','AR','AR','AR','AD'],
                                       ['AU','F','F','F','AD'],
                                       ['AU','AL','F','AD','AD'],
                                       ['AU','AR','AR','AR','G']])
        
        # Now define valid states based on the optimal policy
        self.validStates = [(r,c) for r in range(5) for c in range(5) if self.optimalPolicy[r,c] not in ['F','G']]
        
        # Initialize other attributes
        self.states = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        self.action_symbols = {
            'AU': '\u2191',
            'AD': '\u2193',
            'AL': '\u2190',
            'AR': '\u2192'
        }
        self.actions = list(self.action_symbols.keys())
        self.forbidden_furniture = [(2, 1), (2, 2), (2, 3), (3, 2)]
        self.monsters = [(0, 3), (4, 1)]
        self.food = (4, 4)
        self.gamma = 0.925
        self.delta = 0.1

    def is_valid_state(self, state):
        return (0 <= state[0] < self.rows and
                0 <= state[1] < self.cols and
                state not in self.forbidden_furniture)

    def get_next_state(self, state, action):
        if state == self.food:
            return [(state, 1.0)]

        moves = {
            'AU': (-1, 0),
            'AD': (1, 0),
            'AL': (0, -1),
            'AR': (0, 1)
        }

        probabilities = [0.70, 0.12, 0.12, 0.06]
        intended_move = moves[action]
        possible_moves = [
            intended_move,
            self.get_other(action, 'right'),
            self.get_other(action, 'left'),
            (0, 0)  # Stay in place
        ]

        next_states = []
        for move, prob in zip(possible_moves, probabilities):
            next_state = (state[0] + move[0], state[1] + move[1])
            if self.is_valid_state(next_state):
                next_states.append((next_state, prob))
            else:
                next_states.append((state, prob))

        return next_states

    def get_other(self, action, direction):
        if action in ['AU', 'AD']:
            return (0, 1) if direction == 'right' else (0, -1)
        else:
            return (-1, 0) if direction == 'right' else (1, 0)

    def get_reward(self, state, action, next_state):
        if state == self.food:
            return 0
        elif next_state == self.food:
            return 10
        elif next_state in self.monsters:
            return -8
        else:
            return -0.05
        
    def mcts(self, iterations=50000):
        root = MCTSNode(state=(0, 0))
        action_counts = defaultdict(lambda: defaultdict(int))

        for i in range(iterations):
            epsilon = max(0.01, min(1.0 - i / iterations * 2.0 , 0.1)) # Decaying epsilon
            
            # Start from a random state with epsilon probability
            if random.random() < epsilon:
                random_state = random.choice(self.validStates)
                node = MCTSNode(state=random_state)
                self.expand(node)
            else:
                node = root

            node = self.select(node)
            
            reward = self.simulate(node.state, action_counts)  
            
            # Backpropagate rewards and update visits/wins
            self.backpropagate(node, reward)

            # Update action counts for policy
            if node.parent is not None: 
                for action in node.parent.children.keys():
                    if node.parent.children[action] == node:
                        action_counts[node.parent.state][action] += 1
                        break

        print(f"Visited States: {len(action_counts)} out of {len(self.validStates)}")
        
        # Generate policy from action counts
        policy_dict = {}
        for state in action_counts.keys():
            total_actions = sum(action_counts[state].values())
            policy_dict[state] = {action: count / total_actions for action, count in action_counts[state].items()}

        return policy_dict

    def select(self,node):
         while node.children:
             node=self.best_uct(node)

         # Expand the node if it's not terminal and has been visited
         if not node.children and node.visits > 0:
             self.expand(node)

         return node

    def expand(self,node):
         for action in self.actions:
             if action not in node.children:  
                 next_states=self.get_next_state(node.state ,action)
                 for next_state,_ in next_states:  
                     if next_state not in [child.state for child in node.children.values()]:
                         if self.is_valid_state(next_state):
                             child_node=MCTSNode(state=next_state ,parent=node)
                             node.children[action]=child_node

    def best_action(self ,root):
         if not root.children:  
             raise ValueError("No children available to determine best action.")
        
         best_action=max(root.children.items() ,key=lambda x: x[1].visits)[0]
         return best_action
    
    def simulate(self, state, action_counts):  # Accept action counts as a parameter
        current_state = state
        total_reward = 0
        max_depth = 50
        depth = 0

        while current_state != self.food and current_state not in self.monsters and depth < max_depth:
            possible_actions = [action for action in self.actions if self.get_next_state(current_state, action)]
            
            if not possible_actions:
                break  # No valid actions available
            
            # Epsilon-greedy selection during simulation
            if random.random() < 0.1:  # Exploration
                action = random.choice(possible_actions)
            else:
                action_probs = [action_counts[current_state][action] for action in possible_actions]
                total_probs = sum(action_probs)

                if total_probs == 0:
                    # If total probabilities are zero, fallback to random action
                    action = random.choice(possible_actions)
                else:
                    normalized_probs = [p / total_probs for p in action_probs]
                    action = random.choices(possible_actions, weights=normalized_probs)[0]

            next_states = self.get_next_state(current_state, action)
            
            # Choose the next state based on probabilities
            current_state, prob = max(next_states, key=lambda x: x[1])  
            
            # Calculate the reward for this step
            reward = self.get_reward(state, action, current_state)
            total_reward += reward
            
            depth += 1

        return total_reward if total_reward != 0 else -0.05  # Ensure we return a valid reward
        
    def backpropagate(self,node,reward):
         while node is not None:
             if node.state in self.validStates:
                 node.visits +=1
                 novelty_reward=1 / (1 +node.visits)
                 node.wins +=reward +novelty_reward
             node=node.parent

    def best_uct(self,node):
         best_value=float('-inf')
         best_node=None
        
         for action ,child in node.children.items():
             # Handle the case where visits are zero
             if child.visits == 0:
                 uct_value=float('inf') 
             else:
                 uct_value=(child.wins / child.visits) +np.sqrt(2 *np.log(node.visits) / child.visits)
             
             if uct_value >best_value:
                 best_value=uct_value
                 best_node=child
        
         return best_node

    def printPolicy(self ,policy):
         print("Policy:")
         for r in range(self.rows):
             for c in range(self.cols):
                 if (r,c) in self.forbidden_furniture:
                     print("F\t", end="")
                 elif (r,c) ==self.food:
                     print("G\t", end="")
                 else:
                     actions_probs=policy.get((r,c), {})
                     if actions_probs:
                         best_action_index=max(actions_probs.items(), key=lambda x:x[1])[0]
                         print(f"{best_action_index}\t", end="")
                     else:
                         print("N/A\t", end="") # No valid actions available.
             print()

# Running the MCTS to generate and print the policy.
game=CatVsMonsters()
policy_dict=game.mcts(iterations=100000)
print("Generated Policy:")
game.printPolicy(policy_dict)