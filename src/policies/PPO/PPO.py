import sys
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss

# Fix for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from myenv_5_sparse_reward import MyGridWorld

class PPOTrainer:
    def __init__(self, grid_size=13, total_timesteps=100000, lr=3e-4, 
                 n_steps=2048, batch_size=64, gamma=0.99, ent_coef=0.01, seed=None):
        """
        Initializes the trainer with configurable hyperparameters.
        """
        self.grid_size = grid_size
        self.total_timesteps = total_timesteps
        self.lr = lr
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.seed = seed
        self.model = None
        self.model_path = f"ppo_model_grid{self.grid_size}"

        # Set manual seeds for reproducibility if provided
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    def _setup_env(self):
        """
        Creates and wraps the environment for Stable Baselines3.
        """
        env = MyGridWorld(grid_size=self.grid_size)
        
        # SuperSuit Wrapper Magic
        # Parallel -> Gym Vector Env
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        
        # Concat environments for Parameter Sharing
        env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
        
        # Add monitoring for reward tracking
        env = VecMonitor(env)
        return env

    def train(self):
        """
        Executes the training phase.
        """
        print(f"--- Starting PPO Training (Grid {self.grid_size} | Steps {self.total_timesteps}) ---")
        train_env = self._setup_env()

        self.model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=self.lr,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            gamma=self.gamma,
            ent_coef=self.ent_coef,
            tensorboard_log="./ppo_logs"
        )

        self.model.learn(total_timesteps=self.total_timesteps)
        self.model.save(self.model_path)
        print(f"Model saved at {self.model_path}.zip")

    def test(self, num_test_episodes=100, render=False):
        """
        Tests the trained model and returns metrics (Success rate, Avg Reward, Avg Steps).
        """
        print("\n" + "="*40)
        print("PPO TEST PHASE")
        print(f"Num testing episodes: {num_test_episodes}")
        print("="*40)

        if self.model is None:
            self.model = PPO.load(self.model_path)

        render_mode = "human" if render else None
        # Use clean environment for testing
        test_env = MyGridWorld(grid_size=self.grid_size, render_mode=render_mode)

        success_count = 0
        total_steps = 0
        total_reward = 0

        for episode in range(num_test_episodes):
            observations, _ = test_env.reset(seed=self.seed)
            done = False
            current_episode_reward = 0
            current_steps = 0
            status = "TRUNCATED"

            while not done:
                actions = {}
                for agent_id in test_env.agents:
                    obs = observations[agent_id]
                    # Deterministic actions for testing
                    action, _ = self.model.predict(obs, deterministic=True)
                    actions[agent_id] = action.item() if isinstance(action, np.ndarray) else action

                next_obs, rewards, terminations, truncations, _ = test_env.step(actions)
                observations = next_obs
                current_episode_reward += sum(rewards.values())
                current_steps += 1

                if not test_env.agents or all(terminations.values()) or all(truncations.values()):
                    done = True
                    if any(terminations.values()):
                        status = "SUCCESS"
                        success_count += 1

            total_steps += current_steps
            total_reward += current_episode_reward

            if (episode + 1) % 20 == 0 or episode == 0:
                print(f"Ep {episode+1}/{num_test_episodes}: Status={status}, Rew={current_episode_reward:.1f}, Steps={current_steps}")

        test_env.close()

        # Final Metrics
        success_rate = (success_count / num_test_episodes) * 100
        avg_reward = total_reward / num_test_episodes
        avg_steps = total_steps / num_test_episodes

        print("\n" + "="*40)
        print("TEST METRICS")
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Avg reward: {avg_reward:.2f}")
        print(f"Avg steps: {avg_steps:.2f}")
        print("="*40)

# --- EXECUTION ---
if __name__ == '__main__':
    # You can easily set your parameters here
    trainer = PPOTrainer(
        grid_size=13,
        total_timesteps=100000,
        lr=3e-4,
        ent_coef=0.01,
        seed=1
    )

    # Step 1: Train the model
    trainer.train()

    # Step 2: Evaluate metrics (100 episodes)
    trainer.test(num_test_episodes=100, render=False)
    
    # Step 3: Optional visual render (last few episodes)
    # trainer.test(num_test_episodes=5, render=True)