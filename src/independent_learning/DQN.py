import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

# ----------------------------------------------------------------------
# DQN ALGORITHM CLASS
# ----------------------------------------------------------------------
class DQNAgent:
    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.95, epsilon=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = 64
        
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.memory = deque(maxlen=20000)

    def choose_action(self, observation):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1) # Exploration
        
        with torch.no_grad():
           state = torch.FloatTensor(np.array(observation)).unsqueeze(0).to(self.device) 
           q_values = self.model(state)
           return torch.argmax(q_values).item() # Exploitation

    def store_transition(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones).astype(int)).unsqueeze(1).to(self.device)

        current_q = self.model(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon
    
    
    def save(self, filename):
        base_name = filename
        extension = ".pt"
        
        final_path = f"{base_name}{extension}"
        counter = 1
        while os.path.exists(final_path):
            final_path = f"{base_name}_{counter}{extension}"
            counter += 1

        torch.save(self.model.state_dict(), final_path)
        print(f"Model saved in {final_path}")


    def load(self, filename):
        filename+=".pt"
        if os.path.exists(filename):
            self.model.load_state_dict(torch.load(filename, map_location=self.device))
            print(f"Model loaded from {filename}")

        else:
            print(f"File {filename} not found.")


# ----------------------------------------------------------------------
# TESTING AND TRAINING FUNCTIONS
# ----------------------------------------------------------------------

def train_agents(env, agents, num_episodes, epsilon_decay_rate, min_epsilon):
    """Esegue l'addestramento degli agenti IQL."""
    print("="*40)
    print("IQ TRAINING PHASE")
    print(f"Num training episodes: {num_episodes}\nEspilon decaying rate: {epsilon_decay_rate}")
    print("="*40)
    
    for episode in range(num_episodes):
        observations, _ = env.reset()
        
        # Decaying epsilon
        new_epsilon = max(min_epsilon, agents["agent1"].epsilon * epsilon_decay_rate)
        for agent in agents.values():
            agent.update_epsilon(new_epsilon)
        
        joint_total_reward_episode = 0  
        terminated = {a: False for a in env.possible_agents}
        truncated = {a: False for a in env.possible_agents}
        done = False
        
        while not done:
            actions = {}
            for agent_id in env.agents:
                actions[agent_id] = agents[agent_id].choose_action(observations[agent_id])

            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            joint_total_reward_episode += sum(rewards.values())

            for agent_id in env.agents:
                # 1. Salva l'esperienza nel buffer dell'agente
                agents[agent_id].store_transition(
                    observations[agent_id], 
                    actions[agent_id], 
                    rewards[agent_id], 
                    next_observations[agent_id], 
                    terminations[agent_id]
                )
                # 2. Chiedi all'agente di imparare dal suo buffer
                agents[agent_id].learn()
            observations = next_observations

            if all(terminations.values()) or all(truncations.values()):
                done = True
        
        # Print progression
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}, Epsilon: {agents['agent1'].epsilon:.3f}, Episode joint_total_reward: {joint_total_reward_episode}")

    print("END OF IQ TRAINING PHASE")
    return agents

def test_agents(env, agents, num_test_episodes=100):    
    print("\n" + "="*40)
    print("IQ TEST PHASE")
    print(f"Num testing episodes: {num_test_episodes}")
    print("="*40)
    
    # Epsilon at 0 during test phase, greedy approach
    original_epsilon = agents["agent1"].epsilon
    for agent in agents.values():
        agent.update_epsilon(0.0) 

    success_count = 0
    total_steps = 0
    total_reward = 0
    
    for episode in range(num_test_episodes):
        observations, _ = env.reset()
        done = False
        current_episode_reward = 0
        current_steps = 0
        
        while not done:
            actions = {}
            for agent_id in env.agents:
                actions[agent_id] = agents[agent_id].choose_action(observations[agent_id])

            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            observations = next_observations
            current_episode_reward += sum(rewards.values())
            current_steps += 1
            
            if all(terminations.values()) or all(truncations.values()):
                done = True
                status = "SUCCESS" if all(terminations.values()) else "TRUNCATED"

        total_steps += current_steps
        total_reward += current_episode_reward
        
        
        if all(terminations.values()):
            success_count += 1

        if episode % 20 == 0:
            print(f"Test Ep. {episode+1}/{num_test_episodes}: Status={status}, Reward={current_episode_reward:.2f}, Steps={current_steps}")

    print("END OF TESTING PHASE")
    # Set back epsilon to previous value
    for agent in agents.values():
        agent.update_epsilon(original_epsilon) 

    # Evaluate metrics
    success_rate = (success_count / num_test_episodes) * 100
    avg_reward = total_reward / num_test_episodes
    avg_steps = total_steps / num_test_episodes
    
    print("\n" + "="*40)
    print("TEST METRICS")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Avg reward per episode: {avg_reward:.2f}")
    print(f"Avg steps per episode: {avg_steps:.2f}")
    print("="*40)
    return agents


