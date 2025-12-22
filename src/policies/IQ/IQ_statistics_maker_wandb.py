import sys
import torch
import os
import wandb
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from myenv_5_sparse_reward import MyGridWorld 

# --- CONFIGURATION ---
CONFIG = {
    "algorithm": "SB3_PPO_Shared",
    "grid_size": 13,
    "reward_type": "Sparse",
    "total_timesteps": 100000, 
    "lr": 3e-4,
    "n_steps": 2048, 
    "batch_size": 64,
    "gamma": 0.99,
    "n_seeds": 3,
    "test_interval_steps": 5000, # Eval every X total timesteps
    "n_test_episodes": 10
}

# --- HELPER FOR EVALUATION (Same logic as IQ) ---
def run_evaluation_phase(model, grid_size, n_episodes):
    """Manually runs episodes with the trained model to gather clean metrics."""
    # Create a fresh environment for evaluation
    eval_env = MyGridWorld(grid_size=grid_size)
    
    acc_reward = 0
    acc_steps = 0
    success_count = 0
    
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        curr_reward = 0
        curr_steps = 0
        
        while not done:
            curr_steps += 1
            actions = {}
            for agent_id in eval_env.agents:
                # model.predict returns (action, state)
                action, _ = model.predict(obs[agent_id], deterministic=True)
                actions[agent_id] = action.item()
            
            obs, rewards, terminations, truncations, _ = eval_env.step(actions)
            curr_reward += sum(rewards.values())
            
            if not eval_env.agents or all(terminations.values()) or all(truncations.values()):
                if all(terminations.values()):
                    success_count += 1
                done = True
                
        acc_reward += curr_reward
        acc_steps += curr_steps
    
    eval_env.close()
    return acc_reward/n_episodes, acc_steps/n_episodes, success_count/n_episodes

# --- WANDB CALLBACK WITH INTEGRATED EVALUATION ---
class WandbEvalCallback(BaseCallback):
    def __init__(self, grid_size, test_interval, n_test_episodes, verbose=0):
        super(WandbEvalCallback, self).__init__(verbose)
        self.grid_size = grid_size
        self.test_interval = test_interval
        self.n_test_episodes = n_test_episodes
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        # 1. Log Training rewards (from VecMonitor)
        dones = self.locals['dones']
        infos = self.locals['infos']
        for idx, done in enumerate(dones):
            if done and 'episode' in infos[idx]:
                wandb.log({
                    "train/reward": infos[idx]['episode']['r'],
                    "train/steps": infos[idx]['episode']['l'],
                }, step=self.num_timesteps)

        # 2. Run Periodic Evaluation (Test Phase)
        if self.num_timesteps - self.last_eval_step >= self.test_interval:
            self.last_eval_step = self.num_timesteps
            
            avg_rew, avg_steps, succ_rate = run_evaluation_phase(
                self.model, self.grid_size, self.n_test_episodes
            )
            
            wandb.log({
                "test/avg_reward": avg_rew,
                "test/avg_steps": avg_steps,
                "test/success_rate": succ_rate
            }, step=self.num_timesteps)
            
            print(f"Step {self.num_timesteps} | Eval Success: {succ_rate:.2f} | Rew: {avg_rew:.1f}")
            
        return True

if __name__ == '__main__':
    experiment_group_name = f"{CONFIG['algorithm']}_{CONFIG['grid_size']}_{CONFIG['reward_type']}"
    
    for seed in range(1, CONFIG['n_seeds'] + 1):
        print(f"\n--- SEED {seed}/{CONFIG['n_seeds']} ---")

        run = wandb.init(
            project="MARL_PPO_sparse",
            group=experiment_group_name,
            name=f"seed_{seed}",
            config=CONFIG,
            sync_tensorboard=True,
            reinit=True                  
        )

        np.random.seed(seed)
        torch.manual_seed(seed)

        env = MyGridWorld(grid_size=CONFIG["grid_size"])
        # Wrappers for SB3 compatibility
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
        env = VecMonitor(env)

        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=0, # Terminal output handled by print in callback
            learning_rate=CONFIG['lr'],
            n_steps=CONFIG['n_steps'],
            batch_size=CONFIG['batch_size'],
            gamma=CONFIG['gamma'],
            tensorboard_log=f"runs/{experiment_group_name}"
        )

        # Custom callback that handles both training logs and evaluation
        eval_callback = WandbEvalCallback(
            grid_size=CONFIG["grid_size"],
            test_interval=CONFIG["test_interval_steps"],
            n_test_episodes=CONFIG["n_test_episodes"]
        )

        model.learn(
            total_timesteps=CONFIG["total_timesteps"],
            callback=eval_callback
        )
        
        model.save(f"models/{experiment_group_name}_seed_{seed}")
        run.finish()

    print("\nAll seeds completed.")