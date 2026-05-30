"""
Stability Metrics for Matching Evaluation
Implements DoI, RoI, MD as defined in the paper
"""
import numpy as np
from typing import Dict, List, Tuple, Set
from matching.utils import find_blocking_pairs, get_blocking_agents


class StabilityMetrics:
    """Compute stability metrics for matchings"""
    
    @staticmethod
    def degree_of_instability(matching: Dict[str, str], utilities: Dict[str, np.ndarray],
                             n_agents: int) -> int:
        """
        Degree of Instability (DoI): Number of blocking agents.
        
        Args:
            matching: Current matching
            utilities: Utility matrices
            n_agents: Number of agents per side
        
        Returns:
            Number of agents involved in blocking pairs
        """
        blocking_pairs = find_blocking_pairs(matching, utilities, n_agents)
        if len(blocking_pairs) == 0:
            return 0
        
        blocking_agents = get_blocking_agents(blocking_pairs)
        return len(blocking_agents)
    
    @staticmethod
    def ratio_of_instability(matching: Dict[str, str], utilities: Dict[str, np.ndarray],
                            n_agents: int) -> float:
        """
        Ratio of Instability (RoI): Proportion of blocking pairs.
        RoI = BP(M) / n^2
        
        Args:
            matching: Current matching
            utilities: Utility matrices
            n_agents: Number of agents per side
        
        Returns:
            Proportion of blocking pairs (0 to 1)
        """
        blocking_pairs = find_blocking_pairs(matching, utilities, n_agents)
        total_possible_pairs = n_agents * n_agents
        return len(blocking_pairs) / total_possible_pairs
    
    @staticmethod
    def maximum_dissatisfaction(matching: Dict[str, str], utilities: Dict[str, np.ndarray],
                               n_agents: int) -> float:
        """
        Maximum Dissatisfaction (MD): Maximum utility difference for blocking agents.
        MD(M) = max{U_xv - U_xy} for all blocking pairs (x,v)
        
        Args:
            matching: Current matching
            utilities: Utility matrices
            n_agents: Number of agents per side
        
        Returns:
            Maximum dissatisfaction value
        """
        blocking_pairs = find_blocking_pairs(matching, utilities, n_agents)
        if len(blocking_pairs) == 0:
            return 0.0
        
        U1 = utilities['U1']
        U2 = utilities['U2']
        max_dissatisfaction = 0.0
        
        for i, j in blocking_pairs:
            # For agent i in S1
            current_match_i = matching.get(f's1_{i}')
            if current_match_i is None:
                utility_current = 0.0  # Unmatched
            else:
                current_j = int(current_match_i.split('_')[1])
                utility_current = U1[i, current_j]
            
            utility_blocking = U1[i, j]
            dissatisfaction_i = utility_blocking - utility_current
            max_dissatisfaction = max(max_dissatisfaction, dissatisfaction_i)
            
            # For agent j in S2
            current_match_j = matching.get(f's2_{j}')
            if current_match_j is None:
                utility_current = 0.0
            else:
                current_i = int(current_match_j.split('_')[1])
                utility_current = U2[j, current_i]
            
            utility_blocking = U2[j, i]
            dissatisfaction_j = utility_blocking - utility_current
            max_dissatisfaction = max(max_dissatisfaction, dissatisfaction_j)
        
        return max_dissatisfaction
    
    @staticmethod
    def compute_all_stability_metrics(matching: Dict[str, str], 
                                     utilities: Dict[str, np.ndarray],
                                     n_agents: int) -> Dict[str, float]:
        """
        Compute all stability metrics at once.
        
        Returns:
            Dict with 'is_stable', 'doi', 'roi', 'md', 'num_blocking_pairs'
        """
        blocking_pairs = find_blocking_pairs(matching, utilities, n_agents)
        is_stable = len(blocking_pairs) == 0
        
        if is_stable:
            return {
                'is_stable': True,
                'doi': 0,
                'roi': 0.0,
                'md': 0.0,
                'num_blocking_pairs': 0
            }
        
        # Compute metrics
        blocking_agents = get_blocking_agents(blocking_pairs)
        doi = len(blocking_agents)
        roi = len(blocking_pairs) / (n_agents * n_agents)
        
        # Compute MD
        U1 = utilities['U1']
        U2 = utilities['U2']
        max_dissatisfaction = 0.0
        
        for i, j in blocking_pairs:
            # Agent i dissatisfaction
            current_match_i = matching.get(f's1_{i}')
            utility_current = 0.0 if current_match_i is None else U1[i, int(current_match_i.split('_')[1])]
            dissatisfaction_i = U1[i, j] - utility_current
            max_dissatisfaction = max(max_dissatisfaction, dissatisfaction_i)
            
            # Agent j dissatisfaction
            current_match_j = matching.get(f's2_{j}')
            utility_current = 0.0 if current_match_j is None else U2[j, int(current_match_j.split('_')[1])]
            dissatisfaction_j = U2[j, i] - utility_current
            max_dissatisfaction = max(max_dissatisfaction, dissatisfaction_j)
        
        return {
            'is_stable': False,
            'doi': doi,
            'roi': roi,
            'md': max_dissatisfaction,
            'num_blocking_pairs': len(blocking_pairs)
        }
