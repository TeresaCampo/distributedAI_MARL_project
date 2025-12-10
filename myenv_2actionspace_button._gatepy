import numpy as np
import functools
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
import pygame 
import os
from collections import defaultdict
import random

SPRITES_DIR = "./images_nb"


# Boilerplate PettingZooimport defaultdict 
@functools.lru_cache(maxsize=None)
def observation_space(self, agent): return self.observation_spaces[agent]
@functools.lru_cache(maxsize=None)
def action_space(self, agent): return self.action_spaces[agent]


class VisualGridWorld(ParallelEnv):
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

        # Check if the gate is open or not
        button_pos = self.fixed_components["button"]["pos"]
        gate_pos = self.fixed_components["gate"]["pos"]
        
        desired_positions = {}      
        button_pressed = False
        agents_desire_gate = []
        for agent, action in actions.items():
            current_pos = self.agent_positions[agent].copy()
            target_pos = current_pos.copy()

            if action == 1: target_pos[1] -= 1 
            elif action == 2: target_pos[1] += 1
            elif action == 3: target_pos[0] -= 1
            elif action == 4: target_pos[0] += 1

            # Do not cross walls
            is_wall = self.grid_map[target_pos[0], target_pos[1]] == 1
            is_gate = (target_pos == gate_pos).all()
            is_button = (target_pos == button_pos).all()
            if is_button:
                button_pressed = True
                print("Button pressed")
            if is_gate:
                agents_desire_gate.append((agent, current_pos))
            if is_wall and not is_gate:
                desired_positions[agent] = current_pos 
            
            else:
                desired_positions[agent] = target_pos
                
            # Controllo Ricompense/Terminazioni sui Componenti Fissi nella cella di destinazione
            for name, data in self.fixed_components.items():
                if np.array_equal(target_pos, data["pos"]):
                    rewards[agent] += data["reward"]
                    if name == "gold": terminations[agent] = True
        
        if not button_pressed:
            for agent in agents_desire_gate:
                agent_name = agent[0]
                agent_cur_pos = agent[1]
                desired_positions[agent_name] = agent_cur_pos 

        # Check for conflicts (more than one agents have the same desired position)
        final_positions = self.agent_positions.copy()
        target_counts = defaultdict(list)         # Key: tuple(x, y) of desired positions, Value: list of agents desiring it
        for agent, pos in desired_positions.items():
            pos_tuple = tuple(pos)
            target_counts[pos_tuple].append(agent)
            
        # Solve eventual conflicts
        for pos_tuple, agents_at_target in target_counts.items():      
            # Case 1: no contended position
            if len(agents_at_target) == 1:
                agent = agents_at_target[0]
                final_positions[agent] = desired_positions[agent]
            
            # Case 2: contended position
            else:
                one_agent_already_here_not_moving = None
                for agent in agents_at_target:
                    if self.agent_positions[agent] == pos_tuple:
                        one_agent_already_here_not_moving = agent
                        break
                
                winning_agent = one_agent_already_here_not_moving if one_agent_already_here_not_moving else random.choice(agents_at_target)
                final_positions[winning_agent] = desired_positions[winning_agent]
                
                agents_at_target.remove(winning_agent)
                for losing_agent in agents_at_target[:]:
                    final_positions[losing_agent] = self.agent_positions[losing_agent] 

        self.agent_positions = final_positions
        
        # Observation
        observations = {a: self.agent_positions[a] for a in self.agents}
        # Agents still in the game
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
    
    def generate_test_values(self, agent_num):
        if agent_num==1:
            return[7, 9]
        else:
            return [9,14]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.agent_positions = {
            "agent1": np.array(self.generate_test_values(1)),
            "agent2": np.array(self.generate_test_values(2))
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
    env = VisualGridWorld(render_mode="human")
    observations, infos = env.reset()

    print("Inizio simulazione. Premi Ctrl+C per uscire.")
    
    try:
        # Loop per muovere gli agenti
        for i in range(50): 
            # Azioni casuali per ogni agente attivo
            agent_1 = 1
            agent_2 = 1 if observations["agent2"][1] !=1+3+2 else 0 
            actions = {
                "agent1": agent_1, 
                "agent2": agent_2

            }            
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            if not env.agents:
                print(f"Step {i}: Tutti gli agenti hanno terminato. Reset.")
                env.reset()
            
    except KeyboardInterrupt:
        print("\nSimulazione interrotta dall'utente.")
    finally:
        env.close()