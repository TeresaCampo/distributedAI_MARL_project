import numpy as np
import functools
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
import pygame # <--- Nuova importazione fondamentale

class VisualGridWorld(ParallelEnv):
    # Aggiungiamo 'human' ai modi di render supportati
    metadata = {"render_modes": ["human"], "name": "custom_grid_v0"}

    def __init__(self, render_mode=None):
        # --- Configurazione Logica ---
        self.grid_size = 5
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents[:]
        self.render_mode = render_mode

        # Componenti
        self.fixed_components = {
            "gold":  {"pos": np.array([4, 4]), "reward": 10, "color": (255, 215, 0)}, # Oro
            "trap":  {"pos": np.array([2, 2]), "reward": -5, "color": (255, 0, 0)}    # Rosso
        }
        
        # Mappa (1 = Muro)
        self.grid_map = np.zeros((self.grid_size, self.grid_size))
        self.grid_map[1, 1] = 1
        self.grid_map[1, 2] = 1

        self.action_spaces = {a: spaces.Discrete(5) for a in self.possible_agents}
        self.observation_spaces = {a: spaces.Box(0, self.grid_size-1, shape=(2,), dtype=int) for a in self.possible_agents}

        # --- Configurazione Grafica Pygame ---
        self.window_size = 512  # Dimensione della finestra in pixel
        self.cell_size = self.window_size // self.grid_size # Dimensione di ogni cella
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.agent_positions = {
            "agent_0": np.array([0, 0]),
            "agent_1": np.array([0, 4])
        }
        return {a: self.agent_positions[a] for a in self.agents}, {a: {} for a in self.agents}

    def step(self, actions):
        if not self.agents: return {}, {}, {}, {}, {}
        
        rewards = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        for agent, action in actions.items():
            current_pos = self.agent_positions[agent].copy()
            target_pos = current_pos.copy()

            # Logica Movimento (0=fermo, 1=su, 2=giù, 3=sx, 4=dx)
            # Nota: in matrici numpy, [0] è Y (righe), [1] è X (colonne)
            if action == 1: target_pos[1] -= 1 
            elif action == 2: target_pos[1] += 1
            elif action == 3: target_pos[0] -= 1
            elif action == 4: target_pos[0] += 1
            
            target_pos = np.clip(target_pos, 0, self.grid_size - 1)
            
            if self.grid_map[target_pos[0], target_pos[1]] == 1: # Muro
                target_pos = current_pos 

            self.agent_positions[agent] = target_pos

            for name, data in self.fixed_components.items():
                if np.array_equal(target_pos, data["pos"]):
                    rewards[agent] += data["reward"]
                    if name == "gold": terminations[agent] = True

        observations = {a: self.agent_positions[a] for a in self.agents}
        self.agents = [a for a in self.agents if not terminations[a]]

        # Se siamo in mode human, renderizza ogni step
        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    # --- NUOVO METODO RENDER ---
    def render(self):
        if self.render_mode is None:
            return

        # Inizializza Pygame la prima volta
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255)) # Sfondo Bianco
        
        pix_square_size = self.cell_size 

        # 1. Disegna i Muri (Celle Nere)
        # Nota: Pygame usa (x, y) cartesiani, NumPy usa (row, col). Bisogna invertire o stare attenti.
        # Qui usiamo: x (schermo) = pos[0] * cell_size, y (schermo) = pos[1] * cell_size
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid_map[x, y] == 1:
                    pygame.draw.rect(
                        canvas, 
                        (0, 0, 0), # Colore Nero
                        pygame.Rect(x * pix_square_size, y * pix_square_size, pix_square_size, pix_square_size)
                    )

        # 2. Disegna Componenti (Oro e Trappole)
        for name, data in self.fixed_components.items():
            pos = data["pos"]
            color = data["color"]
            center = (int((pos[0] + 0.5) * pix_square_size), int((pos[1] + 0.5) * pix_square_size))
            
            if name == "gold":
                pygame.draw.circle(canvas, color, center, pix_square_size / 3)
            else:
                # Disegna un quadrato piccolo per la trappola
                rect = pygame.Rect(pos[0] * pix_square_size + 10, pos[1] * pix_square_size + 10, pix_square_size - 20, pix_square_size - 20)
                pygame.draw.rect(canvas, color, rect)

        # 3. Disegna Agenti
        agent_colors = {"agent_0": (0, 0, 255), "agent_1": (0, 200, 0)} # Blu e Verde
        for agent in self.agents:
            pos = self.agent_positions[agent]
            color = agent_colors[agent]
            center = (int((pos[0] + 0.5) * pix_square_size), int((pos[1] + 0.5) * pix_square_size))
            pygame.draw.circle(canvas, color, center, pix_square_size / 2.5)

        # 4. Disegna Griglia (Linee)
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas, 0, (0, pix_square_size * x), (self.window_size, pix_square_size * x), width=2
            )
            pygame.draw.line(
                canvas, 0, (pix_square_size * x, 0), (pix_square_size * x, self.window_size), width=2
            )

        # Aggiorna lo schermo
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(4) # FPS: Controlla la velocità del rendering (4 frame al secondo)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    # Boilerplate PettingZoo
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent): return self.observation_spaces[agent]
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent): return self.action_spaces[agent]


# testo

import time

# Crea l'ambiente in modalità 'human'
env = VisualGridWorld(render_mode="human")
observations, infos = env.reset()

try:
    for _ in range(50): # Esegui 50 passi
        # Azioni casuali
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # Step
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Se tutti hanno finito (preso l'oro), resetta
        if not env.agents:
            env.reset()
            
except KeyboardInterrupt:
    pass
finally:
    env.close()