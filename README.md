# Gridworld cooperative MARL
## Environments overview
In this projects three different environments have been implemented:  
**1. env5**  
One button on the bottom area, it allows agents to open the gate.  
The goal is that all the agents -1 reach the upper area.  
**2. env6**  
Two buttons, one in the bottom and one on the upper area, they allow to open the gate.  
The goal is that all the agents reach the upper area.  
**3. env7**  
Same as env6 but the agents have limited visibility (a radius around them). It is a POMDP.

Both **env 5** and **env 6** have been implemented both with sparse and dense reward.  
**env7** has been implemented with dense reward only.

### Environment implementation
Environment has been implemented basing on [ParallelEnv](https://pettingzoo.farama.org/api/parallel/ ) of PettingZoo.
It implements all the necessary methods and has all the required class variables. The compliance with the ParallelEnv requirements has been tested with the function [test_my_env](./src/test_parallel_env.py) that exploits [testing function provided in PettingZoo documentation](https://pettingzoo.farama.org/content/environment_tests/) .

## Env 5
**env 5** is a simpler scenario, it has been used during the first part of the project in order to learn Environment building skills and RL basis.

<img src="readme_resources/demo_env5.gif" alt="Demo env5" width="350"/>


### Elements
Variable number of agents (in the simulation 2 and 4).
One button, while pressed makes the gate open (self.gate_open = True).
One gate, when open becomes a walkable cell otherwise it behaves as a wall.


### Observation of each agent
*[other_agent_pos * (num_agents-1), button_pos, gate_pos, gate_open]*

Where:
- other_agent_pos, is the relative position (x,y) of another agent
- button_pos, is the relative position (x,y) of the button
- gate_pos, is the relative position (x,y) of the gate
- gate_open, is an int value that represent wheather the gate is open (1) or close (0)

## Env 6
**Env 6** is a more challenging scenario, it has been used to compare two different indipendent learning approaches.

<img src="readme_resources/demo_env6.gif" alt="Demo env6" width="350">


### Elements
Variable number of agents (in the simulation 2 and 4).
Two buttons, while pressed makes the gate open (self.gate_open = True).
One gate, when open becomes a walkable cell otherwise it behaves as a wall.

### Observation of each agent
*[other_agent_pos * (num_agents-1), button_bottom_pos, button_upper_pos, gate_pos, gate_open]*

Where:
- other_agent_pos, is the relative position (x,y) of another agent
- button_bottom_pos, is the relative position (x,y) of the button on bottom area
- button_upper_pos, is the relative position (x,y) of the button on upper area
- gate_pos, is the relative position (x,y) of the gate
- gate_open, is an int value that represent wheather the gate is open (1) or close (0)

## Env7
**Env6** is the hardest scenario because of the partial observability.

### Elements
Variable number of agents (in the simulation 2 and 4).
Two buttons, while pressed makes the gate open (self.gate_open = True).
One gate, when open becomes a walkable cell otherwise it behaves as a wall.

### Observation of each agent
*[other_agent_pos * (num_agents-1), button_bottom_pos, button_upper_pos, gate_pos, gate_open]*

Where:
- other_agent_pos, is the relative position (x,y) of another agent.  
It is equivalent to SENTINEL if the agent is not in the visibility radius.
- button_bottom_pos, is the relative position (x,y) of the button on bottom area.  
It is equivalent to SENTINEL if the button is not in the visibility radius.
- button_upper_pos, is the relative position (x,y) of the button on upper area.  
It is equivalent to SENTINEL if the button is not in the visibility radius.
- gate_pos, is the relative position (x,y) of the gate.  
It is equivalent to SENTINEL if the gate is not in the visibility radius.
- gate_open, is an int value that represent wheather the gate is open (1) or close (0).  
It is equivalent to -1 if the gate is not in the visibility radius.

SENTINEL is an high, infeasible distance, in particular it is equivalent to 2*vision_radius.

## Learning process
In this project I focused on indipendent learning, comparing **strict self play** and **loose self play**.  
In particular I tried three different learning policies: 
- **Indipendent Q-learning**: it worked only with small grids therefore I let it down
- **Indipendent DQN**: it worked olso with bigger grids but was not stable (some iterations worked really well and some others, with the same rewards and parameters, were vacue)
- **Indipendent PPO**: it worked well and was more stable than DQN, I used this policy to evaluate and compare **strict self play** and **loose self play**.  
In particular I exploited [Ray library](https://docs.ray.io/en/latest/rllib/getting-started.html) implementation of PPO.  
I was able to use Ray library to train agents on my custom model wrapping it as a PettingZoo envornment, [following documentation indications](https://docs.ray.io/en/latest/rllib/multi-agent-envs.html#:~:text=PettingZoo%20offers%20a%20repository%20of%20over%2050%20diverse%20multi%2Dagent%20environments%2C%20directly%20compatible%20with%20RLlib%20through%20the%20built%2Din%20PettingZooEnv%20wrapper%3A).


Comparison of the learning processes is visibile in [this file](/src/strictplay_looseplay_comparison/comparison.ipynb).  
The plots have been generated using files present in [this folder](/src/policies/PPO/).
# To try it on your pc
First install the required dependencies
- **Solution 1**: :

    Install dependencies from requirements.txt

    These requirements match with Python 3.10.19
    ```bash
    pip install requirements.txt
    ```
- **Solution 2**

    Create conda environment with environment.yml
    ```bash
    conda env create -f environment.yml
    ```

