"""
Preference Generation for Stable Matching Variants
Implements SM, SMI, and SMT with symmetric/asymmetric preferences.

Paper spec: "Uij is picked uniformly from range [k,l] in Z,
while maintaining the strict preference order."
- SM/SMT: range [1, 10]
- SMI: range [-10, 10] (negative = unacceptable)

Symmetric: pi(j) = pj(i) and Uij = Uji
  -> unique pair values guarantee exactly 1 stable matching
Asymmetric: random values maintaining strict order per agent
  -> random gaps help agents discriminate through noisy rewards
"""
import numpy as np
from typing import Dict, Optional


def _random_strict_order(rng, n: int, low: float, high: float) -> np.ndarray:
    """
    Generate n unique random values in [low, high] in descending order.
    Assigns them to a random permutation of agents.
    Returns array of length n where array[j] = utility for agent j.
    """
    vals = np.sort(rng.uniform(low, high, n))[::-1]  # descending
    perm = rng.permutation(n)
    result = np.zeros(n, dtype=np.float32)
    for rank, j in enumerate(perm):
        result[j] = vals[rank]
    return result


class PreferenceGenerator:

    @staticmethod
    def generate_sm(n_agents: int, symmetric: bool = True,
                    seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        SM: weights from U[1,10] with strict ordering.
        Symmetric: Uij = Uji, unique pair values -> exactly 1 stable matching.
        Asymmetric: random values per agent maintaining strict order.
        """
        rng = np.random.default_rng(seed)

        if symmetric:
            # Assign unique random values to each unordered pair (i,j)
            # This guarantees pi(j)=pj(i) and exactly 1 stable matching
            U = np.zeros((n_agents, n_agents), dtype=np.float32)
            n_pairs = n_agents * (n_agents - 1) // 2
            raw = rng.permutation(n_pairs) + 1
            scaled = 1.0 + (raw - 1) * 9.0 / max(n_pairs - 1, 1)  # [1,10]
            idx = 0
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    U[i, j] = scaled[idx]
                    U[j, i] = scaled[idx]
                    idx += 1
            return {'U1': U, 'U2': U.T}

        else:
            U1 = np.zeros((n_agents, n_agents), dtype=np.float32)
            U2 = np.zeros((n_agents, n_agents), dtype=np.float32)
            for i in range(n_agents):
                U1[i] = _random_strict_order(rng, n_agents, 1.0, 10.0)
            for j in range(n_agents):
                U2[j] = _random_strict_order(rng, n_agents, 1.0, 10.0)
            return {'U1': U1, 'U2': U2}

    @staticmethod
    def generate_smi(n_agents: int, incomplete_ratio: float = 0.3,
                     symmetric: bool = True,
                     seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        SMI: weights from U[-10,10]. Negative = unacceptable.
        Acceptable agents get random positive values in strict order.
        Unacceptable agents get random negative values.
        """
        rng = np.random.default_rng(seed)
        n_unacceptable = max(1, int(n_agents * incomplete_ratio))

        def _make_smi_row(rng, n, n_unacceptable):
            row = np.zeros(n, dtype=np.float32)
            unacceptable = rng.choice(n, n_unacceptable, replace=False)
            acceptable = [j for j in range(n) if j not in unacceptable]
            # Negative utilities for unacceptable
            for j in unacceptable:
                row[j] = float(rng.uniform(-10, 0))
            # Random positive values in strict order for acceptable
            if acceptable:
                vals = np.sort(rng.uniform(1, 10, len(acceptable)))[::-1]
                perm = rng.permutation(len(acceptable))
                for rank, idx in enumerate(perm):
                    row[acceptable[idx]] = vals[rank]
            return row

        if symmetric:
            U1 = np.zeros((n_agents, n_agents), dtype=np.float32)
            for i in range(n_agents):
                U1[i] = _make_smi_row(rng, n_agents, n_unacceptable)
            return {'U1': U1, 'U2': U1.T}
        else:
            U1 = np.zeros((n_agents, n_agents), dtype=np.float32)
            U2 = np.zeros((n_agents, n_agents), dtype=np.float32)
            for i in range(n_agents):
                U1[i] = _make_smi_row(rng, n_agents, n_unacceptable)
            for j in range(n_agents):
                U2[j] = _make_smi_row(rng, n_agents, n_unacceptable)
            return {'U1': U1, 'U2': U2}

    @staticmethod
    def generate_smt(n_agents: int, tie_probability: float = 0.2,
                     symmetric: bool = True,
                     seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        SMT: weights from U[1,10], ties allowed.
        Ties created by repeating previous value with tie_probability.
        """
        rng = np.random.default_rng(seed)

        def _make_smt_row(rng, n, tie_probability):
            vals = []
            for _ in range(n):
                if vals and rng.random() < tie_probability:
                    vals.append(vals[-1])
                else:
                    vals.append(float(rng.uniform(1, 10)))
            sorted_vals = sorted(vals, reverse=True)
            # Assign to random permutation of agents
            perm = rng.permutation(n)
            row = np.zeros(n, dtype=np.float32)
            for rank, j in enumerate(perm):
                row[j] = sorted_vals[rank]
            return row

        if symmetric:
            U1 = np.zeros((n_agents, n_agents), dtype=np.float32)
            for i in range(n_agents):
                U1[i] = _make_smt_row(rng, n_agents, tie_probability)
            return {'U1': U1, 'U2': U1.T}
        else:
            U1 = np.zeros((n_agents, n_agents), dtype=np.float32)
            U2 = np.zeros((n_agents, n_agents), dtype=np.float32)
            for i in range(n_agents):
                U1[i] = _make_smt_row(rng, n_agents, tie_probability)
            for j in range(n_agents):
                U2[j] = _make_smt_row(rng, n_agents, tie_probability)
            return {'U1': U1, 'U2': U2}
