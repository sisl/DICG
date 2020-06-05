# Using local gym
import sys
import os
current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path + '/../../')


from envs.ma_gym.envs.predator_prey import PredatorPrey
import gym
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageColor

from .utils import standard_eval

PRE_IDS = {
    'agent': 'A',
    'prey': 'P',
    'wall': 'W',
    'empty': '0'
}

class PredatorPreyWrapper(PredatorPrey):

    def __init__(self, centralized, other_agent_visible=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._agent_visible = other_agent_visible
        self.action_space = self.action_space[0]
        if self._agent_visible:
            mask_size = np.prod(self._agent_view_mask)
            self._obs_low = np.array([0., 0.] + [0.] * mask_size * 2 + [0.0])
            self._obs_high = np.array([1., 1.] + [1.] * mask_size * 2 + [1.0])
            self.observation_space = gym.spaces.Box(self._obs_low, self._obs_high)
        else:
            self.observation_space = self.observation_space[0]
        self.centralized = centralized
        if centralized:
            self.observation_space = gym.spaces.Box(
                low=np.array(list(self.observation_space.low) * self.n_agents),
                high=np.array(list(self.observation_space.high) * self.n_agents)
            )
        self.pickleable = True

    def get_avail_actions(self):
        avail_actions = [[1] * self.action_space.n for _ in range(self.n_agents)]
        if not self.centralized:
            return avail_actions
        else:
            return np.concatenate(avail_actions)

    def get_agent_obs(self):
        obs = super().get_agent_obs()
        if self._agent_visible:
            for i_agent in range(self.n_agents):
                pos = self.agent_pos[i_agent]
                # check if other agents are in the view area
                _agent_pos = np.zeros(self._agent_view_mask)
                for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
                    for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
                        if PRE_IDS['agent'] in self._full_obs[row][col]:
                            # get relative position for the prey loc:
                            _agent_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = 1

                obs[i_agent].extend(_agent_pos.flatten().tolist())
        return obs

    def step(self, actions):
        obses, rewards, dones, infos = super().step(actions)
        if not self.centralized:
            return obses, rewards, dones, infos
        else:
            return np.concatenate(obses), np.mean(rewards), np.all(dones), infos

    def reset(self):
        obses = super().reset()
        if not self.centralized:
            return obses
        else:
            return np.concatenate(obses)

    def eval(self, policy, n_episodes=20, greedy=True, load_from_file=False, 
             render=False):
        standard_eval(self, policy, n_episodes=n_episodes, greedy=greedy, 
            load_from_file=load_from_file, render=render)

    def my_render(self, attention_weights=None, cell_size=35):
        if attention_weights is not None:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            img = Image.fromarray(self.render(mode='rgb_array'))
            
            # Plot attention weights for agent 0

            start_col, start_row = self.agent_pos[0]
            start_x, start_y = (start_row + 0.5) * cell_size, (start_col + 0.5) * cell_size
            for i in range(self.n_agents):
                if i == 0:
                    draw_circle(
                        img, self.agent_pos[i], cell_size=cell_size, 
                        fill=None, outline='green', radius=0.1,
                        width=int(20 * attention_weights[i]))
                else:
                    if attention_weights[i] == 0:
                        fill = None
                    else:
                        fill = 'green'
                    end_col, end_row = self.agent_pos[i]
                    end_x, end_y = (end_row + 0.5) * cell_size, (end_col + 0.5) * cell_size
                    ImageDraw.Draw(img).line(((start_x, start_y), (end_x, end_y)), 
                        fill=fill, width=int(20 * attention_weights[i]))

            img = np.asarray(img)
            self.viewer.imshow(img)
            return self.viewer.isopen
        else:
            self.render(mode='human')

    def is_neighbor(self, i, j):
        pos_i = self.agent_pos[i]
        pos_j = self.agent_pos[j]
        if np.abs(pos_i[0] - pos_j[0]) <= 1 and np.abs(pos_i[1] - pos_j[1]) <= 1:
            return True
        else:
            return False

    def get_proximal_adjacency_matrix(self, obs_n, dec_obs_dim):
        obs_n_original_shape = obs_n.shape
        obs_n = obs_n.reshape(-1, self.n_agents, dec_obs_dim)
        self_connected_adj = \
            torch.zeros(obs_n.shape[:-2] + (self.n_agents, self.n_agents))
        D = torch.zeros_like(self_connected_adj)
        
        if len(obs_n.shape) == 2:
            for i_agent in range(self.n_agents):
                # Self connection
                self_connected_adj[i_agent, i_agent] = 1
                D[i_agent, i_agent] += 1
                for j_agent in range(i_agent + 1, self.n_agents):
                    if self.is_neighbor(i_agent, j_agent):
                        self_connected_adj[i_agent, j_agent] = 1
                        self_connected_adj[j_agent, i_agent] = 1
                        D[i_agent, i_agent] += 1
                        D[j_agent, j_agent] += 1

            D = D.diagonal().pow(-0.5).diag()

        else:
            for i_step in range(int(obs_n.shape[0])):
                for i_agent in range(self.n_agents):
                    # Self connection
                    self_connected_adj[i_step, i_agent, i_agent] = 1
                    D[i_step, i_agent, i_agent] += 1
                    for j_agent in range(i_agent + 1, self.n_agents):
                        if self.is_neighbor(i_agent, j_agent):
                            self_connected_adj[i_step, i_agent, j_agent] = 1
                            self_connected_adj[i_step, j_agent, i_agent] = 1
                            D[i_step, i_agent, i_agent] += 1
                            D[i_step, j_agent, j_agent] += 1
                
                D[i_step, :, :] = D[i_step, :, :].diagonal().pow(-0.5).diag()

            self_connected_adj = self_connected_adj.reshape(
                obs_n_original_shape[:-2] + (self.n_agents, self.n_agents))
            D = D.reshape(
                obs_n_original_shape[:-2] + (self.n_agents, self.n_agents))

            processed_self_connected_adj = torch.matmul(D, self_connected_adj)
            processed_self_connected_adj = torch.matmul(processed_self_connected_adj, D)

        return self_connected_adj, processed_self_connected_adj



def draw_circle(image, pos, cell_size=50, fill='white', outline='black', 
                radius=0.3, width=1):
    col, row = pos
    row, col = row * cell_size, col * cell_size
    gap = cell_size * radius
    x, y = row + gap, col + gap
    x_dash, y_dash = row + cell_size - gap, col + cell_size - gap
    ImageDraw.Draw(image).ellipse([(x, y), (x_dash, y_dash)], fill=fill, 
        outline=outline, width=width)


if __name__ == '__main__':
    env = PredatorPreyWrapper(centralized=True)
    print(env.action_space)
    print(env.observation_space)