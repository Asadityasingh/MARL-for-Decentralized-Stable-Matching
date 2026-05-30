"""
Grid-based Matching Environment for Decentralized Stable Matching
Implements the spatial formulation from the paper
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional


class MatchingEnvironment(gym.Env):
    """
    Grid world environment for two-sided stable matching.
    
    Agents from two disjoint sets (S1, S2) navigate a grid to find matches.
    Matching occurs when two agents from opposite sets are in the same cell
    and both express interest in matching with each other.
    """
    
    def __init__(self, grid_size: Tuple[int, int], n_agents_per_side: int, 
                 utilities: Dict[str, np.ndarray], seed: Optional[int] = None):
        """
        Args:
            grid_size: (rows, cols) dimensions of the grid
            n_agents_per_side: number of agents in each set (S1 and S2)
            utilities: dict with 'U1' and 'U2' utility matrices
                      U1[i,j] = utility for agent i in S1 matching with agent j in S2
                      U2[j,i] = utility for agent j in S2 matching with agent i in S1
            seed: random seed for reproducibility
        """
        super().__init__()
        
        self.rows, self.cols = grid_size
        self.n_agents = n_agents_per_side
        self.utilities = utilities
        self.rng = np.random.default_rng(seed)
        
        # Agent positions (fixed starting positions per episode)
        self.start_positions_s1 = None
        self.start_positions_s2 = None
        self.positions_s1 = None  # Current positions
        self.positions_s2 = None
        
        # Matching state
        self.matches = {}  # {agent_id: matched_agent_id or None}
        self.interest_s1 = {}  # {agent_id: target_agent_id or None}
        self.interest_s2 = {}
        
        # Observation and action spaces (will be set per agent)
        self.obs_dim = self.rows * self.cols + 2 * self.n_agents
        self.action_dim = 4 + self.n_agents  # 4 moves + n matching actions
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Initialize random starting positions (fixed for episode)
        if self.start_positions_s1 is None:
            self.start_positions_s1 = self._random_positions(self.n_agents)
            self.start_positions_s2 = self._random_positions(self.n_agents)
        
        # Reset to starting positions
        self.positions_s1 = self.start_positions_s1.copy()
        self.positions_s2 = self.start_positions_s2.copy()
        
        # Clear matches and interests
        self.matches = {f's1_{i}': None for i in range(self.n_agents)}
        self.matches.update({f's2_{i}': None for i in range(self.n_agents)})
        self.interest_s1 = {i: None for i in range(self.n_agents)}
        self.interest_s2 = {i: None for i in range(self.n_agents)}
        
        # Get initial observations
        observations = self._get_observations()
        info = {}
        
        return observations, info
    
    def step(self, actions: Dict[str, int]):
        """
        Execute one time step.
        
        Args:
            actions: dict mapping agent_id to action
                    action in [0,1,2,3] = move [up, down, left, right]
                    action in [4, 4+n_agents) = express interest in agent (action-4)
        
        Returns:
            observations, rewards, terminated, truncated, info
        """
        # Process movement actions
        self._process_movements(actions)
        
        # Process matching interests
        self._process_interests(actions)
        
        # Update matches based on mutual interest in same cell
        self._update_matches()
        
        # Compute rewards
        rewards = self._compute_rewards()
        
        # Get observations
        observations = self._get_observations()
        
        # Episode doesn't terminate (continuous learning)
        terminated = {agent: False for agent in actions.keys()}
        terminated['__all__'] = False
        truncated = {agent: False for agent in actions.keys()}
        truncated['__all__'] = False
        
        info = {'matches': self.matches.copy()}
        
        return observations, rewards, terminated, truncated, info
    
    def _random_positions(self, n: int) -> np.ndarray:
        """Generate random positions on grid"""
        positions = np.zeros((n, 2), dtype=int)
        for i in range(n):
            positions[i] = [
                self.rng.integers(0, self.rows),
                self.rng.integers(0, self.cols)
            ]
        return positions
    
    def _process_movements(self, actions: Dict[str, int]):
        """Process movement actions (0-3)"""
        for agent_id, action in actions.items():
            if action < 4:  # Movement action
                if agent_id.startswith('s1_'):
                    idx = int(agent_id.split('_')[1])
                    pos = self.positions_s1[idx]
                else:
                    idx = int(agent_id.split('_')[1])
                    pos = self.positions_s2[idx]
                
                # Apply movement (clip to grid boundaries)
                if action == 0:  # Up
                    pos[0] = max(0, pos[0] - 1)
                elif action == 1:  # Down
                    pos[0] = min(self.rows - 1, pos[0] + 1)
                elif action == 2:  # Left
                    pos[1] = max(0, pos[1] - 1)
                elif action == 3:  # Right
                    pos[1] = min(self.cols - 1, pos[1] + 1)
    
    def _process_interests(self, actions: Dict[str, int]):
        """Process matching interest actions (4+)"""
        for agent_id, action in actions.items():
            if action >= 4:  # Matching action
                target_idx = action - 4
                if agent_id.startswith('s1_'):
                    idx = int(agent_id.split('_')[1])
                    self.interest_s1[idx] = target_idx
                else:
                    idx = int(agent_id.split('_')[1])
                    self.interest_s2[idx] = target_idx
            else:
                # No interest expressed
                if agent_id.startswith('s1_'):
                    idx = int(agent_id.split('_')[1])
                    self.interest_s1[idx] = None
                else:
                    idx = int(agent_id.split('_')[1])
                    self.interest_s2[idx] = None
    
    def _update_matches(self):
        """Update matches based on mutual interest in same cell — vectorized."""
        for agent_id in self.matches:
            self.matches[agent_id] = None

        s1_cells = self.positions_s1[:, 0] * self.cols + self.positions_s1[:, 1]
        s2_cells = self.positions_s2[:, 0] * self.cols + self.positions_s2[:, 1]
        same_cell = (s1_cells[:, None] == s2_cells[None, :])  # (n, n)

        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if same_cell[i, j] and self.interest_s1[i] == j and self.interest_s2[j] == i:
                    self.matches[f's1_{i}'] = f's2_{j}'
                    self.matches[f's2_{j}'] = f's1_{i}'
    
    def _compute_rewards(self) -> Dict[str, float]:
        """Compute noisy rewards for all agents"""
        rewards = {}
        
        for i in range(self.n_agents):
            agent_id = f's1_{i}'
            if self.matches[agent_id] is None:
                rewards[agent_id] = -1.0
            else:
                j = int(self.matches[agent_id].split('_')[1])
                # Noisy reward: U_ij * C, where C ~ N(1, 0.1)
                noise = self.rng.normal(1.0, 0.1)
                rewards[agent_id] = self.utilities['U1'][i, j] * noise
        
        for j in range(self.n_agents):
            agent_id = f's2_{j}'
            if self.matches[agent_id] is None:
                rewards[agent_id] = -1.0
            else:
                i = int(self.matches[agent_id].split('_')[1])
                noise = self.rng.normal(1.0, 0.1)
                rewards[agent_id] = self.utilities['U2'][j, i] * noise
        
        return rewards
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents — fully vectorized."""
        n = self.n_agents
        grid_size = self.rows * self.cols

        # Precompute cell indices for all agents
        s1_cells = self.positions_s1[:, 0] * self.cols + self.positions_s1[:, 1]  # (n,)
        s2_cells = self.positions_s2[:, 0] * self.cols + self.positions_s2[:, 1]  # (n,)

        # Precompute colocation matrices: same_cell[i,j] = True if s1_i and s2_j in same cell
        same_cell = (s1_cells[:, None] == s2_cells[None, :])  # (n, n)

        # Precompute interest arrays
        interest_s1_arr = np.array([self.interest_s1[i] if self.interest_s1[i] is not None else -1
                                    for i in range(n)], dtype=np.int32)
        interest_s2_arr = np.array([self.interest_s2[j] if self.interest_s2[j] is not None else -1
                                    for j in range(n)], dtype=np.int32)

        observations = {}

        # S1 agents: obs = [pos_onehot(grid_size), s2_in_cell(n), s2_interested_in_me(n)]
        for i in range(n):
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            obs[s1_cells[i]] = 1.0
            # which s2 agents are in same cell
            in_cell = same_cell[i]  # (n,) bool
            obs[grid_size: grid_size + n] = in_cell.astype(np.float32)
            # which of those s2 agents are interested in me (i)
            interested = in_cell & (interest_s2_arr == i)
            obs[grid_size + n: grid_size + 2*n] = interested.astype(np.float32)
            observations[f's1_{i}'] = obs

        # S2 agents: obs = [pos_onehot(grid_size), s1_in_cell(n), s1_interested_in_me(n)]
        for j in range(n):
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            obs[s2_cells[j]] = 1.0
            # which s1 agents are in same cell
            in_cell = same_cell[:, j]  # (n,) bool
            obs[grid_size: grid_size + n] = in_cell.astype(np.float32)
            # which of those s1 agents are interested in me (j)
            interested = in_cell & (interest_s1_arr == j)
            obs[grid_size + n: grid_size + 2*n] = interested.astype(np.float32)
            observations[f's2_{j}'] = obs

        return observations
    
    def render(self):
        """Render the environment (optional)"""
        grid = np.zeros((self.rows, self.cols), dtype=str)
        grid[:] = '.'
        
        for i in range(self.n_agents):
            r, c = self.positions_s1[i]
            grid[r, c] = f'1{i}' if grid[r, c] == '.' else grid[r, c] + f',1{i}'
        
        for j in range(self.n_agents):
            r, c = self.positions_s2[j]
            grid[r, c] = f'2{j}' if grid[r, c] == '.' else grid[r, c] + f',2{j}'
        
        print("\nGrid:")
        for row in grid:
            print(' '.join(f'{cell:8s}' for cell in row))
        print(f"\nMatches: {self.matches}")
