"""
Matching Utilities
Helper functions for analyzing and validating matchings
"""
import numpy as np
from typing import Dict, List, Tuple, Set


def matching_dict_to_pairs(matching: Dict[str, str], n_agents: int) -> List[Tuple[int, int]]:
    """
    Convert matching dict to list of (i, j) pairs where i in S1, j in S2.
    
    Args:
        matching: dict mapping agent_id to matched_agent_id
        n_agents: number of agents per side
    
    Returns:
        List of (i, j) tuples
    """
    pairs = []
    for i in range(n_agents):
        agent_id = f's1_{i}'
        if matching.get(agent_id) is not None:
            matched_id = matching[agent_id]
            j = int(matched_id.split('_')[1])
            pairs.append((i, j))
    return pairs


def get_preference_order(utilities: np.ndarray) -> np.ndarray:
    """
    Convert utility matrix to preference order (rank).
    Higher utility = lower rank (better).
    
    Args:
        utilities: utility matrix
    
    Returns:
        Rank matrix where rank[i,j] is the rank of j in i's preference
    """
    n = utilities.shape[0]
    ranks = np.zeros_like(utilities, dtype=int)
    
    for i in range(n):
        # Sort indices by utility (descending)
        sorted_indices = np.argsort(-utilities[i])
        for rank, j in enumerate(sorted_indices):
            ranks[i, j] = rank + 1  # 1-indexed
    
    return ranks


def find_blocking_pairs(matching: Dict[str, str], utilities: Dict[str, np.ndarray],
                       n_agents: int) -> List[Tuple[int, int]]:
    """
    Find all blocking pairs in a matching.
    
    A pair (i, j) is blocking if:
    - i prefers j to current match
    - j prefers i to current match
    
    Args:
        matching: current matching
        utilities: utility matrices
        n_agents: number of agents per side
    
    Returns:
        List of blocking pairs (i, j)
    """
    U1 = utilities['U1']
    U2 = utilities['U2']
    blocking_pairs = []
    
    for i in range(n_agents):
        current_match_i = matching.get(f's1_{i}')
        
        for j in range(n_agents):
            current_match_j = matching.get(f's2_{j}')
            
            # Check if (i, j) is a blocking pair
            if current_match_i is None:
                # i is unmatched, prefers j if U1[i,j] > 0 (for SMI)
                i_prefers_j = U1[i, j] > 0
            else:
                # i prefers j if U1[i,j] > U1[i, current_j]
                current_j = int(current_match_i.split('_')[1])
                i_prefers_j = U1[i, j] > U1[i, current_j]
            
            if current_match_j is None:
                # j is unmatched, prefers i if U2[j,i] > 0
                j_prefers_i = U2[j, i] > 0
            else:
                # j prefers i if U2[j,i] > U2[j, current_i]
                current_i = int(current_match_j.split('_')[1])
                j_prefers_i = U2[j, i] > U2[j, current_i]
            
            if i_prefers_j and j_prefers_i:
                blocking_pairs.append((i, j))
    
    return blocking_pairs


def is_stable(matching: Dict[str, str], utilities: Dict[str, np.ndarray],
              n_agents: int) -> bool:
    """Check if matching is stable (no blocking pairs)"""
    blocking_pairs = find_blocking_pairs(matching, utilities, n_agents)
    return len(blocking_pairs) == 0


def get_blocking_agents(blocking_pairs: List[Tuple[int, int]]) -> Set[str]:
    """Get set of agent IDs involved in blocking pairs"""
    agents = set()
    for i, j in blocking_pairs:
        agents.add(f's1_{i}')
        agents.add(f's2_{j}')
    return agents
