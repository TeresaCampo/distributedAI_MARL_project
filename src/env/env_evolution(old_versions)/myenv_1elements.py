import numpy as np
import functools
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
import pygame 
import os

SPRITES_DIR = "./sprites"

# Boilerplate PettingZoo
@functools.lru_cache(maxsize=None)
def observation_space(self, agent): return self.observation_spaces[agent]
@functools.lru_cache(maxsize=None)
def action_space(self, agent): return self.action_spaces[agent]


class MyGridWorld(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "custom_grid_v0"}

    def __init__(self, render_mode=None):
        self.grid_size = 15
        self.render_mode = render_mode

        ######## Agents, fixed components, and forbidden positions
        self.possible_agents = ["agent1", "agent2"]
        self.agents = self.possible_agents[:]

        self.fixed_components = {
            "button":  {"pos": np.array([self.grid_size//2+2, 1+3+2]), "reward": 10, "file": "button.png"},
            "gate":  {"pos": np.array([self.grid_size//2, 1+3]), "reward": -5, "file": "gate.png"}   
        }

        self.x_range = (1, self.grid_size-1-1)
        self.y_range = (1+3+1, self.grid_size-1-1)
        self.forbidden_position = self.fixed_components["button"]
    
        ######## Pygame graphic configuration  
        self.window_size = 810
        self.cell_size = self.window_size // self.grid_size
        self.window = None
        self.clock = None
        
        # Walls
        self.grid_map = np.zeros((self.grid_size, self.grid_size))
        self.grid_map[0, :] = 1
        self.grid_map[:, 0] = 1
        self.grid_map[-1, :] = 1
        self.grid_map[:, -1] = 1
        self.grid_map[ :, 1+3] = 1

        # Sprites 
        self.agent_sprites = {}
        self.component_sprites = {}

        ######## Action space and observation space
        # Five possible actions in each grid: stay(0), up(1), down(2), left(3), right(4)
        self.action_spaces = {a: spaces.Discrete(5) for a in self.possible_agents}
        self.observation_spaces = {a: spaces.Box(0, self.grid_size-1, shape=(2,), dtype=int) for a in self.possible_agents}

    '''
    Step in the environment
    '''
    def step(self, actions):
        if not self.agents: return {}, {}, {}, {}, {}
        
        rewards = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        for agent, action in actions.items():
            current_pos = self.agent_positions[agent].copy()
            target_pos = current_pos.copy()

            # Five possible actions in each grid: stay(0), up(1), down(2), left(3), right(4)
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

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos
   
    '''
    Determines initial condition of the simulation
    '''
    def generate_valid_position(self):
        while True:
            x = np.random.randint(self.x_range)
            y = np.random.randint(self.y_range)
            new_position = (y, x)

            if new_position not in self.forbidden_position:
                return np.array([x, y])
                
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.agent_positions = {
            "agent1": np.array([6, 1+3+5]),
            "agent2": np.array([8, 1+3+8])
        }
        
        # To visualize the reset position if "human" mode on
        if self.render_mode == "human":
            self.render() 

        return {a: self.agent_positions[a] for a in self.agents}, {a: {} for a in self.agents}
     
    '''
    Functions for Sprite rendering and loading
    '''    
    def _create_missing_sprite(self, text, bg_color, border_color = (0,0,0)):
        
        error_sprite = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        error_sprite.fill((0, 0, 0, 0))
        center = (self.cell_size // 2, self.cell_size // 2)
        radius = int(self.cell_size * 0.4)
        pygame.draw.circle(error_sprite, bg_color, center, radius) 
        pygame.draw.circle(error_sprite, border_color, center, radius, 2)
        
        # Add text label
        try:
            font_size = int(self.cell_size * 0.22)
            font = pygame.font.Font(None, font_size)   
            text_color = (0, 0, 0) if sum(bg_color) > 300 else (255, 255, 255) 
            text_surface = font.render(text, True, text_color)
            text_rect = text_surface.get_rect(center=center)
            error_sprite.blit(text_surface, text_rect)
            
        except Exception as e:
            print(f"WARNING: Error while writing text inside missing sprite {text}. A missing sprite without text label will be used for this simulation.")
            pass
            
        return error_sprite
    
    def _load_and_scale_sprite(self, filename, component_name):
        path = os.path.join(SPRITES_DIR, filename)
    
        try:
            image = pygame.image.load(path).convert_alpha() 
            scaled_size = int(self.cell_size )
            return pygame.transform.scale(image, (scaled_size, scaled_size))
            
        except (FileNotFoundError, pygame.error) as e:
            print(f"WARNING! Sprite image {path} not found. Using default sprite for this simulation.")
    
            text = component_name.upper() if component_name else "ERROR"
            bg_color = (100, 100, 255) if "AGENT" in text else (128,128,128) # Blue agent, Gray fixed components
            return self._create_missing_sprite(text, bg_color)

    def render(self):
        if self.render_mode is None:
            return

        if self.window is None:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            
            for agent_name in self.agents:
                self.agent_sprites[agent_name] = self._load_and_scale_sprite(f"{agent_name}.png", agent_name)
            for name, data in self.fixed_components.items():
                self.component_sprites[name] = self._load_and_scale_sprite(data["file"], name)
                 
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((220, 220, 220)) 
        pix_square_size = self.cell_size 
        sprite_offset = (pix_square_size - int(pix_square_size * 0.8)) // 2

        # Render walls
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid_map[x, y] == 1:
                    pygame.draw.rect(
                        canvas, 
                        (50, 50, 50), # Grigio scuro per i muri
                        pygame.Rect(x * pix_square_size, y * pix_square_size, pix_square_size, pix_square_size)
                    )

        # Render fixed components and agents
        for name, data in self.fixed_components.items():
            pos = data["pos"]
            x_coord = pos[0] * pix_square_size + sprite_offset
            y_coord = pos[1] * pix_square_size + sprite_offset

            sprite = self.component_sprites[name]
            canvas.blit(sprite, (x_coord, y_coord))

        for agent in self.agents:
            pos = self.agent_positions[agent]            
            x_coord = pos[0] * pix_square_size + sprite_offset
            y_coord = pos[1] * pix_square_size + sprite_offset
            
            sprite = self.agent_sprites[agent]
            canvas.blit(sprite, (x_coord, y_coord))

        # Render the grid
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas, 0, (0, pix_square_size * x), (self.window_size, pix_square_size * x), width=2
            )
            pygame.draw.line(
                canvas, 0, (pix_square_size * x, 0), (pix_square_size * x, self.window_size), width=2
            )

        # Display the render
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(4) # FPS

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()




# --- Esempio di Esecuzione ---
if __name__ == '__main__':
    env = MyGridWorld(render_mode="human")
    observations, infos = env.reset()

    print("Inizio simulazione. Premi Ctrl+C per uscire.")
    
    try:
        # Loop per muovere gli agenti
        for i in range(50): 
            # Azioni casuali per ogni agente attivo
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            if not env.agents:
                print(f"Step {i}: Tutti gli agenti hanno terminato. Reset.")
                env.reset()
            
    except KeyboardInterrupt:
        print("\nSimulazione interrotta dall'utente.")
    finally:
        env.close()