import sys
import os
import numpy as np
import random
import wandb  

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(str(parent_dir) + "/independent_learning")

# --- MODIFICA 1: Importiamo DQNAgent invece di IQ ---
from DQN import DQNAgent 
from myenv_5_dense_reward import MyGridWorld


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def run_single_episode(env, agents, is_training=True):
    """
    Runs a single episode adapted for DQN.
    """
    observations, _ = env.reset()
    terminations = {a: False for a in env.possible_agents}
    truncations = {a: False for a in env.possible_agents}
    done = False
    
    total_reward = 0
    steps = 0
    
    while not done:
        steps += 1
        actions = {}
        
        # 1. Action Selection (solo agenti vivi)
        for agent_id in env.agents:
            actions[agent_id] = agents[agent_id].choose_action(observations[agent_id])

        if not actions: 
            break

        # 2. Step Environment
        next_observations, rewards, terminations, truncations, infos = env.step(actions)
        total_reward += sum(rewards.values())

        # 3. Learning (Only if training)
        if is_training:
            for agent_id in env.agents:
                agent = agents[agent_id]
                
                # Gestione next_state per agenti terminati
                real_next_state = next_observations.get(agent_id, observations[agent_id])
                is_terminated = terminations.get(agent_id, False)

                agent.store_transition(
                    observations[agent_id], 
                    actions[agent_id], 
                    rewards[agent_id], 
                    real_next_state, 
                    is_terminated
                )
                agent.learn()

        observations = next_observations

        # Check termination
        if not env.agents or all(terminations.values()) or all(truncations.values()):
            done = True
            
    success = all(terminations.get(a, False) for a in env.possible_agents)
    return total_reward, steps, success

def run_evaluation_phase(env, agents, n_episodes):
    """Runs N test episodes with Epsilon=0 and returns the averages."""
    
    # 1. Save current epsilon and set to 0 for testing (Greedy)
    saved_epsilons = {aid: agent.epsilon for aid, agent in agents.items()}
    for agent in agents.values():
        agent.update_epsilon(0.0)
        
    success_count = 0
    acc_reward = 0
    acc_steps = 0
    
    # 2. Test episodes loop
    for _ in range(n_episodes):
        rew, stp, succ = run_single_episode(env, agents, is_training=False)
        acc_reward += rew
        acc_steps += stp
        if succ: success_count += 1
        
    # 3. Restore original epsilon for training
    for aid, agent in agents.items():
        agent.update_epsilon(saved_epsilons[aid])
        
    # 4. Calculate metrics
    avg_reward = acc_reward / n_episodes
    avg_steps = acc_steps / n_episodes
    success_rate = success_count / n_episodes
    return avg_reward, avg_steps, success_rate

# ---------------------------------------------------------
# MAIN EXECUTION LOOP
# ---------------------------------------------------------

# --- Configuration ---
CONFIG = {
    "algorithm": "Independent_DQN", # Aggiornato nome algoritmo
    "grid_size": 13,
    "reward_type": "Dense",
    "n_seeds": 3,
    "n_episodes": 5000,
    "test_interval": 50,
    "n_test_episodes": 10,
    "epsilon_decay": 0.9997,
    "min_epsilon": 0.01,
    "lr": 0.001,  # DQN usa spesso LR più bassi (es. 0.001 o 0.0001) rispetto a IQ
    "gamma": 0.9,
    "initial_epsilon": 1.0
}

if __name__ == '__main__':
    experiment_group_name = f"{CONFIG['algorithm']}_{CONFIG['grid_size']}_{CONFIG['reward_type']}"

    for seed in range(1, CONFIG["n_seeds"] + 1):
        print(f"\n--- Starting Seed {seed}/{CONFIG['n_seeds']} ---")
        
        run = wandb.init(
            project="MARL_DQN_dense",
            group=experiment_group_name,
            name=f"seed_{seed}",
            config=CONFIG,
            reinit=True
        )

        # Setup Env e Agenti
        env = MyGridWorld(grid_size=CONFIG["grid_size"])
        agents = {}
        action_size = env.action_space("agent1").n
        obs_shape = env.observation_space("agent1").shape[0]

        random.seed(seed)
        np.random.seed(seed)
        # Importante per DQN e PyTorch: settare seed anche per torch se possibile
        # (Opzionale: torch.manual_seed(seed))

        for agent_id in env.possible_agents:
            # --- MODIFICA: Istanziazione DQNAgent ---
            agents[agent_id] = DQNAgent(
                obs_dim=obs_shape,          # DQN.py si aspetta obs_dim
                action_dim=action_size,     # DQN.py si aspetta action_dim
                lr=CONFIG["lr"], 
                gamma=CONFIG["gamma"],
                epsilon=CONFIG["initial_epsilon"]
            )

        # --- TRAINING LOOP ---
        for episode in range(1, CONFIG["n_episodes"] + 1):
            
            # Epsilon Decay
            current_eps = agents["agent1"].epsilon
            new_epsilon = max(CONFIG["min_epsilon"], current_eps * CONFIG["epsilon_decay"])
            for agent in agents.values():
                agent.update_epsilon(new_epsilon)
            
            # Train step
            train_reward, train_steps, train_success = run_single_episode(env, agents, is_training=True)

            # Train LOG
            wandb.log({
                "episode": episode,
                "train/reward": train_reward,
                "train/steps": train_steps,
                "train/epsilon": new_epsilon,
                "train/success": int(train_success)
            }, step=episode)

            # --- EVALUATION LOOP ---
            if episode % CONFIG["test_interval"] == 0:
                avg_rew, avg_steps, succ_rate = run_evaluation_phase(env, agents, CONFIG["n_test_episodes"])
                
                # Test LOG
                wandb.log({
                    "episode": episode,
                    "test/avg_reward": avg_rew,
                    "test/avg_steps": avg_steps,
                    "test/success_rate": succ_rate
                }, step=episode)
                
                print(f"Ep {episode} | Test Rate: {succ_rate:.2f} | Rew: {avg_rew:.1f}")
        run.finish()

    print("THE END.")