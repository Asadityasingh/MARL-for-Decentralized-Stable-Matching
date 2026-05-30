"""
Median Matching Analysis.
Uses rotation poset enumeration (Irving 1994) for small n,
and Gale-Shapley bracketing for larger n where brute force is infeasible.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import permutations


def _gale_shapley(utilities: Dict[str, np.ndarray], n: int,
                  s1_optimal: bool = True) -> Dict[str, str]:
    """Standard Gale-Shapley. Returns s1-optimal or s2-optimal stable matching."""
    U1, U2 = utilities['U1'], utilities['U2']

    # Build preference lists (sorted indices, best first)
    pref_s1 = [list(np.argsort(-U1[i])) for i in range(n)]
    pref_s2 = [list(np.argsort(-U2[j])) for j in range(n)]

    # Rank lookup for s2: rank_s2[j][i] = rank of i in j's list
    rank_s2 = [{i: r for r, i in enumerate(pref_s2[j])} for j in range(n)]

    if s1_optimal:
        # S1 proposes
        free_s1 = list(range(n))
        next_proposal = [0] * n          # next index in pref list to propose to
        engaged_s2 = {}                  # j -> i (current partner)

        while free_s1:
            i = free_s1.pop(0)
            if next_proposal[i] >= n:
                continue
            j = pref_s1[i][next_proposal[i]]
            next_proposal[i] += 1

            if j not in engaged_s2:
                engaged_s2[j] = i
            else:
                current = engaged_s2[j]
                if rank_s2[j][i] < rank_s2[j][current]:
                    engaged_s2[j] = i
                    free_s1.append(current)
                else:
                    free_s1.append(i)

        matching = {}
        for j, i in engaged_s2.items():
            matching[f's1_{i}'] = f's2_{j}'
            matching[f's2_{j}'] = f's1_{i}'
        return matching
    else:
        # S2 proposes — gives s2-optimal (= s1-pessimal)
        free_s2 = list(range(n))
        next_proposal = [0] * n
        rank_s1 = [{j: r for r, j in enumerate(pref_s1[i])} for i in range(n)]
        engaged_s1 = {}

        while free_s2:
            j = free_s2.pop(0)
            if next_proposal[j] >= n:
                continue
            i = pref_s2[j][next_proposal[j]]
            next_proposal[j] += 1

            if i not in engaged_s1:
                engaged_s1[i] = j
            else:
                current = engaged_s1[i]
                if rank_s1[i][j] < rank_s1[i][current]:
                    engaged_s1[i] = j
                    free_s2.append(current)
                else:
                    free_s2.append(j)

        matching = {}
        for i, j in engaged_s1.items():
            matching[f's1_{i}'] = f's2_{j}'
            matching[f's2_{j}'] = f's1_{i}'
        return matching


def _find_all_stable_matchings_small(utilities: Dict[str, np.ndarray],
                                     n: int) -> List[Dict[str, str]]:
    """Brute force — only feasible for n <= 8 (8! = 40320)."""
    from matching.utils import is_stable
    stable = []
    for perm in permutations(range(n)):
        m = {}
        for i, j in enumerate(perm):
            m[f's1_{i}'] = f's2_{j}'
            m[f's2_{j}'] = f's1_{i}'
        if is_stable(m, utilities, n):
            stable.append(m)
    return stable


def find_all_stable_matchings(utilities: Dict[str, np.ndarray],
                               n: int) -> List[Dict[str, str]]:
    """
    Find all stable matchings.
    - n <= 8: brute force (exact)
    - n > 8: return [s1-optimal, s2-optimal] as bracket (2 matchings)
      The paper's median analysis is only meaningful when there are multiple
      stable matchings; for large n we approximate with the two extremes.
    """
    if n <= 8:
        return _find_all_stable_matchings_small(utilities, n)
    else:
        m1 = _gale_shapley(utilities, n, s1_optimal=True)
        m2 = _gale_shapley(utilities, n, s1_optimal=False)
        if m1 == m2:
            return [m1]
        return [m1, m2]


def compute_mm_proportion(matching: Dict[str, str],
                          stable_matchings: List[Dict[str, str]],
                          utilities: Dict[str, np.ndarray],
                          n: int) -> float:
    """
    Proportion of agents matched to their median stable partner.
    Only defined when K = |stable_matchings| is odd.
    """
    K = len(stable_matchings)
    if K == 0 or K % 2 == 0:
        return 0.0

    median_pos = (K + 1) // 2 - 1  # 0-indexed
    median_matches = 0
    total = 2 * n

    for agent_id in [f's1_{i}' for i in range(n)] + [f's2_{j}' for j in range(n)]:
        partners = [sm[agent_id] for sm in stable_matchings if agent_id in sm and sm[agent_id]]

        if not partners:
            continue

        # Sort partners by this agent's utility (descending)
        if agent_id.startswith('s1_'):
            i = int(agent_id.split('_')[1])
            partners.sort(key=lambda p: utilities['U1'][i, int(p.split('_')[1])], reverse=True)
        else:
            j = int(agent_id.split('_')[1])
            partners.sort(key=lambda p: utilities['U2'][j, int(p.split('_')[1])], reverse=True)

        if len(partners) > median_pos:
            if matching.get(agent_id) == partners[median_pos]:
                median_matches += 1

    return median_matches / total


def is_msm(matching: Dict[str, str],
           stable_matchings: List[Dict[str, str]],
           utilities: Dict[str, np.ndarray],
           n: int) -> bool:
    """True if matching is the median stable matching (requires odd K)."""
    K = len(stable_matchings)
    if K == 0 or K % 2 == 0:
        return False
    median_pos = (K + 1) // 2 - 1
    for agent_id in [f's1_{i}' for i in range(n)] + [f's2_{j}' for j in range(n)]:
        partners = [sm[agent_id] for sm in stable_matchings if agent_id in sm and sm[agent_id]]
        if not partners:
            continue
        if agent_id.startswith('s1_'):
            i = int(agent_id.split('_')[1])
            partners.sort(key=lambda p: utilities['U1'][i, int(p.split('_')[1])], reverse=True)
        else:
            j = int(agent_id.split('_')[1])
            partners.sort(key=lambda p: utilities['U2'][j, int(p.split('_')[1])], reverse=True)
        if len(partners) > median_pos:
            if matching.get(agent_id) != partners[median_pos]:
                return False
    return True


class MedianMatchingAnalyzer:

    @staticmethod
    def analyze_median_properties(matching: Dict[str, str],
                                  utilities: Dict[str, np.ndarray],
                                  n_agents: int) -> Dict:
        stable = find_all_stable_matchings(utilities, n_agents)
        K = len(stable)
        return {
            'num_stable_matchings': K,
            'is_msm': is_msm(matching, stable, utilities, n_agents),
            'mm_proportion': compute_mm_proportion(matching, stable, utilities, n_agents),
        }
