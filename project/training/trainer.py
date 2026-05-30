"""
Training Loop for Multi-Agent SARSA with WandB integration.
Streams per-episode metrics to WandB for real-time monitoring.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import json
import wandb


class Trainer:
    """Multi-agent SARSA trainer with WandB logging."""

    def __init__(self, env, agents: Dict, config: Dict,
                 wandb_run: Optional[wandb.sdk.wandb_run.Run] = None):
        self.env = env
        self.agents = agents
        self.config = config
        self.wandb_run = wandb_run

        self.n_episodes = config['n_episodes']
        self.steps_per_episode = config['steps_per_episode']
        self.batch_size = config.get('batch_size', 32)
        self.log_frequency = config.get('log_frequency', 1000)
        self.save_frequency = config.get('save_frequency', 10000)

        # History for stats
        self.episode_rewards: Dict[str, List[float]] = {aid: [] for aid in agents}
        self.episode_matches: List[Dict] = []
        self.losses: Dict[str, List[float]] = {aid: [] for aid in agents}

    def train(self, save_dir: str = None) -> Dict:
        save_path = Path(save_dir) if save_dir else None
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)

        n_agents_per_side = len(self.agents) // 2

        for episode in range(self.n_episodes):
            ep_rewards, ep_losses = self._run_episode()

            # Update networks + decay epsilon
            for aid, agent in self.agents.items():
                loss = agent.update(self.batch_size)
                ep_losses[aid] = loss if loss else 0.0
                self.losses[aid].append(ep_losses[aid])
                agent.decay_epsilon()

            # Store rewards
            for aid in self.agents:
                self.episode_rewards[aid].append(np.mean(ep_rewards[aid]))

            # --- WandB logging every 500 episodes ---
            # 128 parallel runs causes rate limit on free WandB tier
            if self.wandb_run and episode % 500 == 0:
                matched_pairs = sum(
                    1 for v in self.episode_matches[-1].values() if v is not None
                ) // 2
                epsilon = list(self.agents.values())[0].epsilon
                avg_reward = np.mean([np.mean(ep_rewards[aid]) for aid in self.agents])
                avg_loss = np.mean(list(ep_losses.values()))

                s1_reward = np.mean([np.mean(ep_rewards[aid])
                                     for aid in self.agents if aid.startswith('s1_')])
                s2_reward = np.mean([np.mean(ep_rewards[aid])
                                     for aid in self.agents if aid.startswith('s2_')])

                metrics = {
                    'train/episode': episode,
                    'train/avg_reward': avg_reward,
                    'train/s1_avg_reward': s1_reward,
                    'train/s2_avg_reward': s2_reward,
                    'train/matched_pairs': matched_pairs,
                    'train/match_rate': matched_pairs / n_agents_per_side,
                    'train/epsilon': epsilon,
                    'train/avg_loss': avg_loss,
                }

                # Moving averages at key intervals
                if episode >= 100:
                    recent = [np.mean(self.episode_rewards[aid][-100:])
                              for aid in self.agents]
                    metrics['train/reward_ma100'] = np.mean(recent)
                if episode >= 1000:
                    recent = [np.mean(self.episode_rewards[aid][-1000:])
                              for aid in self.agents]
                    metrics['train/reward_ma1000'] = np.mean(recent)

                self.wandb_run.log(metrics, step=episode)

            # Console log
            if episode % self.log_frequency == 0:
                matched_pairs = sum(
                    1 for v in self.episode_matches[-1].values() if v is not None
                ) // 2
                epsilon = list(self.agents.values())[0].epsilon
                avg_reward = np.mean([np.mean(ep_rewards[aid]) for aid in self.agents])
                print(f"  Ep {episode:6d}/{self.n_episodes} | "
                      f"Reward: {avg_reward:7.3f} | "
                      f"Matched: {matched_pairs}/{n_agents_per_side} | "
                      f"ε: {epsilon:.4f}")

            # Checkpoint
            if save_path and episode % self.save_frequency == 0 and episode > 0:
                self._save_checkpoint(save_path, episode)

        if save_path:
            self._save_final(save_path)

        return self._get_training_stats()

    def _run_episode(self):
        obs, _ = self.env.reset()
        ep_rewards = {aid: [] for aid in self.agents}
        ep_losses = {aid: 0.0 for aid in self.agents}

        actions = {aid: self.agents[aid].select_action(obs[aid]) for aid in self.agents}

        for _ in range(self.steps_per_episode):
            next_obs, rewards, terminated, _, info = self.env.step(actions)
            next_actions = {aid: self.agents[aid].select_action(next_obs[aid])
                            for aid in self.agents}

            for aid in self.agents:
                self.agents[aid].store_transition(
                    obs[aid], actions[aid], rewards[aid],
                    next_obs[aid], next_actions[aid], terminated[aid]
                )
                ep_rewards[aid].append(rewards[aid])

            obs = next_obs
            actions = next_actions

        for agent in self.agents.values():
            agent.end_episode()

        self.episode_matches.append(info['matches'].copy())
        return ep_rewards, ep_losses

    def evaluate(self, n_episodes: int = 10) -> Dict:
        """Greedy evaluation (epsilon=0). Returns matches for metric computation."""
        eval_rewards = {aid: [] for aid in self.agents}
        eval_matches = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            ep_rewards = {aid: [] for aid in self.agents}
            actions = {aid: self.agents[aid].select_action(obs[aid], training=False)
                       for aid in self.agents}

            for _ in range(self.steps_per_episode):
                next_obs, rewards, _, _, info = self.env.step(actions)
                next_actions = {aid: self.agents[aid].select_action(next_obs[aid], training=False)
                                for aid in self.agents}
                for aid in self.agents:
                    ep_rewards[aid].append(rewards[aid])
                obs = next_obs
                actions = next_actions

            for aid in self.agents:
                eval_rewards[aid].append(np.mean(ep_rewards[aid]))
            eval_matches.append(info['matches'].copy())

        return {
            'avg_reward': {aid: float(np.mean(eval_rewards[aid])) for aid in self.agents},
            'std_reward': {aid: float(np.std(eval_rewards[aid])) for aid in self.agents},
            'matches': eval_matches,
        }

    def _save_checkpoint(self, save_path: Path, episode: int):
        ckpt_dir = save_path / f'checkpoint_{episode}'
        ckpt_dir.mkdir(exist_ok=True)
        for aid, agent in self.agents.items():
            agent.save(str(ckpt_dir / f'{aid}.pt'))

    def _save_final(self, save_path: Path):
        models_dir = save_path / 'final_models'
        models_dir.mkdir(exist_ok=True)
        for aid, agent in self.agents.items():
            agent.save(str(models_dir / f'{aid}.pt'))

        logs_dir = save_path / 'logs'
        logs_dir.mkdir(exist_ok=True)
        pd.DataFrame(self.episode_rewards).to_csv(logs_dir / 'episode_rewards.csv', index=False)
        pd.DataFrame(self.losses).to_csv(logs_dir / 'losses.csv', index=False)
        with open(logs_dir / 'final_matching.json', 'w') as f:
            json.dump(self.episode_matches[-1], f, indent=2)

    def _get_training_stats(self) -> Dict:
        return {
            'final_avg_reward': {
                aid: float(np.mean(self.episode_rewards[aid][-100:]))
                for aid in self.agents
            },
            'final_matching': self.episode_matches[-1] if self.episode_matches else {},
        }
