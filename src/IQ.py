import random
from collections import defaultdict
import numpy as np


# ----------------------------------------------------------------------
# IQ ALGORITHM CLASS
# ----------------------------------------------------------------------
class IQLearningAgent:
    def __init__(self, agent_id, action_space_size, obs_space_shape, learning_rate=0.1, discount_factor=0.9, epsilon=1.0):
        self.id = agent_id
        self.action_space_size = action_space_size
        self.epsilon = epsilon
        self.lr = learning_rate
        self.gamma = discount_factor
        
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))

    
    def _state_to_key(self, observation):
        return tuple(observation.flatten())

    def choose_action(self, observation):
        state_key = self._state_to_key(observation)
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)  # Exploration
        else:
            return np.argmax(self.q_table[state_key])  # Exploitation

    def learn(self, state, action, reward, next_state, terminated):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
 
        current_q = self.q_table[state_key][action]
        if terminated:
            max_future_q = 0
        else:
            max_future_q = np.max(self.q_table[next_state_key])

        # Bellman equation update
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[state_key][action] = new_q

    def update_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon
  

def train_agents(env, agents, num_episodes, epsilon_decay_rate, min_epsilon):
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
            # Learning phase
            for agent_id in env.agents:
                agents[agent_id].learn(
                    state=observations[agent_id], 
                    action=actions[agent_id], 
                    reward=rewards[agent_id], 
                    next_state=next_observations[agent_id], 
                    terminated=terminations[agent_id]
                )
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
    
    print("END OF IQ TESTING PHASE")
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
