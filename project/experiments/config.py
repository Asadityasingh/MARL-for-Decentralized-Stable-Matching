"""
Experiment Configuration — exact paper specs.
Epsilon decay is computed so epsilon reaches epsilon_min at 80% of training.
Formula: decay = (epsilon_min / epsilon_start) ^ (1 / (0.8 * n_episodes))
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import json


def compute_epsilon_decay(n_episodes: int,
                          epsilon_start: float = 1.0,
                          epsilon_min: float = 0.05) -> float:
    """Decay rate so epsilon hits epsilon_min at 80% of training."""
    return float((epsilon_min / epsilon_start) ** (1.0 / (0.8 * n_episodes)))


@dataclass
class ExperimentConfig:
    name: str
    grid_size: Tuple[int, int]
    n_agents: int
    problem_type: str       # 'SM', 'SMI', 'SMT'
    symmetric: bool
    n_episodes: int
    steps_per_episode: int
    n_instances: int = 10   # paper: 10 instances per config
    n_runs: int = 5         # paper: 5 runs per instance

    # Paper hyperparameters (Section 5)
    lr: float = 1e-4
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    batch_size: int = 32
    replay_buffer_size: int = 10  # last 10 episodes

    # Computed from n_episodes — do not set manually
    epsilon_decay: float = field(init=False)

    # Logging
    log_frequency: int = 1000
    save_frequency: int = 10000

    def __post_init__(self):
        self.epsilon_decay = compute_epsilon_decay(
            self.n_episodes, self.epsilon_start, self.epsilon_min
        )

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'grid_size': list(self.grid_size),
            'n_agents': self.n_agents,
            'problem_type': self.problem_type,
            'symmetric': self.symmetric,
            'n_episodes': self.n_episodes,
            'steps_per_episode': self.steps_per_episode,
            'n_instances': self.n_instances,
            'n_runs': self.n_runs,
            'lr': self.lr,
            'gamma': self.gamma,
            'epsilon_start': self.epsilon_start,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'replay_buffer_size': self.replay_buffer_size,
        }

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def get_all_experiment_configs() -> List[ExperimentConfig]:
    """
    All 54 configurations from the paper.
    Grid: 3x3, 4x4, 5x5
    Agents: 8 (3x3 only), 8/10/12/14 (4x4, 5x5)
    Types: SM, SMI, SMT  x  symmetric, asymmetric
    Episodes: 100k-400k depending on complexity
    Steps: 300-700 depending on grid/agents
    """
    configs = []

    # 3x3: only 8 agents
    # sym: 100k episodes (unique stable matching, converges fast)
    # asym: 200k episodes (multiple stable matchings, needs more time)
    for problem_type in ['SM', 'SMI', 'SMT']:
        for symmetric in [True, False]:
            sym_str = 'sym' if symmetric else 'asym'
            n_eps = 100000 if symmetric else 200000
            configs.append(ExperimentConfig(
                name=f'3x3_8agents_{problem_type}_{sym_str}',
                grid_size=(3, 3),
                n_agents=8,
                problem_type=problem_type,
                symmetric=symmetric,
                n_episodes=n_eps,
                steps_per_episode=300,
            ))

    # 4x4: 8/10/12/14 agents
    agent_episodes_4x4 = {8: 100000, 10: 200000, 12: 300000, 14: 300000}
    agent_steps_4x4 = {8: 400, 10: 400, 12: 500, 14: 500}
    for n_agents in [8, 10, 12, 14]:
        for problem_type in ['SM', 'SMI', 'SMT']:
            for symmetric in [True, False]:
                sym_str = 'sym' if symmetric else 'asym'
                configs.append(ExperimentConfig(
                    name=f'4x4_{n_agents}agents_{problem_type}_{sym_str}',
                    grid_size=(4, 4),
                    n_agents=n_agents,
                    problem_type=problem_type,
                    symmetric=symmetric,
                    n_episodes=agent_episodes_4x4[n_agents],
                    steps_per_episode=agent_steps_4x4[n_agents],
                ))

    # 5x5: 8/10/12/14 agents
    agent_episodes_5x5 = {8: 100000, 10: 200000, 12: 400000, 14: 400000}
    agent_steps_5x5 = {8: 500, 10: 500, 12: 700, 14: 700}
    for n_agents in [8, 10, 12, 14]:
        for problem_type in ['SM', 'SMI', 'SMT']:
            for symmetric in [True, False]:
                sym_str = 'sym' if symmetric else 'asym'
                configs.append(ExperimentConfig(
                    name=f'5x5_{n_agents}agents_{problem_type}_{sym_str}',
                    grid_size=(5, 5),
                    n_agents=n_agents,
                    problem_type=problem_type,
                    symmetric=symmetric,
                    n_episodes=agent_episodes_5x5[n_agents],
                    steps_per_episode=agent_steps_5x5[n_agents],
                ))

    return configs


def get_validation_config() -> ExperimentConfig:
    """
    Single config to validate the full pipeline before full run.
    Paper's easiest case: 3x3, 8 agents, SM, symmetric → expects 100% stability.
    Uses 2 instances x 2 runs to keep it fast (~30 min on MPS).
    """
    return ExperimentConfig(
        name='validate_3x3_8agents_SM_sym',
        grid_size=(3, 3),
        n_agents=8,
        problem_type='SM',
        symmetric=True,
        n_episodes=100000,
        steps_per_episode=300,
        n_instances=2,
        n_runs=2,
    )
