import sys
import os
import numpy as np
import random
from collections import defaultdict
import wandb  

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(str(parent_dir) + "/independent_learning")

from IQ import IQLearningAgent 
from myenv_5_sparse_reward import MyGridWorld


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def run_single_episode(env, agents, is_training=True):
    """
    Runs a single episode.
    If is_training=True: performs the Q-Learning update.
    If is_training=False: executes action only (Epsilon must be handled externally).
    """
    observations, _ = env.reset()
    terminated = {a: False for a in env.possible_agents}
    truncations = {a: False for a in env.possible_agents}
    done = False
    
    total_reward = 0
    steps = 0
    
    while not done:
        steps += 1
        actions = {}
        
        # 1. Action Selection
        for agent_id in env.agents:
            actions[agent_id] = agent.choose_action(observations[agent_id])

        # 2. Step Environment
        next_observations, rewards, terminations, truncations, infos = env.step(actions)
        total_reward += sum(rewards.values())

        # 3. Learning (Only if training)
        if is_training:
            for agent_id in env.agents:
                agent.learn(
                    state=observations[agent_id], 
                    action=actions[agent_id], 
                    reward=rewards[agent_id], 
                    next_state=next_observations[agent_id], 
                    terminated=terminations[agent_id]
                )

        observations = next_observations

        # Check termination
        if all(terminations.values()) or all(truncations.values()):
            done = True
            
    success = all(terminations.values())
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
    "algorithm": "Independent_IQ",
    "grid_size": 15,
    "reward_type": "Sparse",
    "n_seeds": 3,
    "n_episodes": 5000,
    "test_interval": 50,
    "n_test_episodes": 10,
    "epsilon_decay": 0.9995,
    "min_epsilon": 0.01,
    "lr": 0.1,
    "gamma": 0.9,
    "initial_epsilon": 1.0
}

if __name__ == '__main__':
    
    # Questo nome di gruppo unisce le run dei diversi seed sotto un unico grafico
    experiment_group_name = f"{CONFIG['algorithm']}_{CONFIG['grid_size']}_{CONFIG['reward_type']}"

    print(f"Starting Experiment: {experiment_group_name}")

    for seed in range(1, CONFIG["n_seeds"] + 1):
        print(f"\n--- Starting Seed {seed}/{CONFIG['n_seeds']} ---")
        
        # 1. Inizializza W&B per QUESTO seed
        run = wandb.init(
            project="MARL_Project_GridWorld", # Nome del tuo progetto sulla dashboard
            group=experiment_group_name,      # FONDAMENTALE: Raggruppa i 3 seed insieme
            name=f"seed_{seed}",              # Nome specifico di questa linea
            config=CONFIG,                    # Salva i parametri così sai cosa hai lanciato
            reinit=True                       # Permette di lanciare più run nello stesso script
        )

        # Setup Env e Agenti
        env = MyGridWorld(grid_size=CONFIG["grid_size"])
        agents = {}
        action_size = env.action_space("agent1").n
        obs_shape = env.observation_space("agent1").shape[0]

        random.seed(seed)
        np.random.seed(seed)

        for agent_id in env.possible_agents:
            agents[agent_id] = IQLearningAgent(
                agent_id=agent_id, 
                action_space_size=action_size, 
                obs_space_shape=obs_shape,
                learning_rate=CONFIG["lr"], 
                discount_factor=CONFIG["gamma"],
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

            # LOGGING SU WANDB (Train)
            # Usiamo un prefisso "train/" per separare i grafici
            wandb.log({
                "episode": episode,
                "train/reward": train_reward,
                "train/steps": train_steps,
                "train/epsilon": new_epsilon,
                "train/success": int(train_success)
            }, step=episode) # 'step' allinea l'asse X

            # --- EVALUATION LOOP ---
            if episode % CONFIG["test_interval"] == 0:
                avg_rew, avg_steps, succ_rate = run_evaluation_phase(env, agents, CONFIG["n_test_episodes"])
                
                # LOGGING SU WANDB (Test)
                wandb.log({
                    "episode": episode,
                    "test/avg_reward": avg_rew,
                    "test/avg_steps": avg_steps,
                    "test/success_rate": succ_rate
                }, step=episode)
                
                print(f"Ep {episode} | Test Rate: {succ_rate:.2f} | Rew: {avg_rew:.1f}")

        # Chiudi la run corrente su W&B prima di iniziare il prossimo seed
        run.finish()

    print("Experiment completed. Check charts at wandb.ai")