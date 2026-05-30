"""
Fairness Metrics for Matching Evaluation
Implements regret cost, egalitarian cost, set-equality cost
"""
import numpy as np
from typing import Dict, List, Tuple
from matching.utils import get_preference_order, matching_dict_to_pairs


class FairnessMetrics:
    """Compute fairness metrics for matchings"""
    
    @staticmethod
    def regret_cost(matching: Dict[str, str], utilities: Dict[str, np.ndarray],
                   n_agents: int) -> int:
        """
        Regret Cost: r(M) = max max{pi(j), pj(i)} for all (i,j) in M
        Maximum rank among all matched pairs.
        
        Args:
            matching: Current matching
            utilities: Utility matrices
            n_agents: Number of agents per side
        
        Returns:
            Maximum rank (lower is better)
        """
        U1 = utilities['U1']
        U2 = utilities['U2']
        
        # Get preference orders (ranks)
        ranks_s1 = get_preference_order(U1)
        ranks_s2 = get_preference_order(U2)
        
        max_rank = 0
        for i in range(n_agents):
            match = matching.get(f's1_{i}')
            if match is not None:
                j = int(match.split('_')[1])
                rank_i = ranks_s1[i, j]
                rank_j = ranks_s2[j, i]
                max_rank = max(max_rank, rank_i, rank_j)
        
        return max_rank
    
    @staticmethod
    def egalitarian_cost(matching: Dict[str, str], utilities: Dict[str, np.ndarray],
                        n_agents: int) -> int:
        """
        Egalitarian Cost: c(M) = Σ(pi(j) + pj(i)) for all (i,j) in M
        Sum of all ranks.
        
        Args:
            matching: Current matching
            utilities: Utility matrices
            n_agents: Number of agents per side
        
        Returns:
            Sum of ranks (lower is better)
        """
        U1 = utilities['U1']
        U2 = utilities['U2']
        
        ranks_s1 = get_preference_order(U1)
        ranks_s2 = get_preference_order(U2)
        
        total_cost = 0
        for i in range(n_agents):
            match = matching.get(f's1_{i}')
            if match is not None:
                j = int(match.split('_')[1])
                total_cost += ranks_s1[i, j] + ranks_s2[j, i]
        
        return total_cost
    
    @staticmethod
    def set_equality_cost(matching: Dict[str, str], utilities: Dict[str, np.ndarray],
                         n_agents: int) -> int:
        """
        Set-Equality Cost: d(M) = |Σpi(j) - Σpj(i)| for all (i,j) in M
        Absolute difference between sum of ranks for each side.
        
        Args:
            matching: Current matching
            utilities: Utility matrices
            n_agents: Number of agents per side
        
        Returns:
            Absolute difference (lower is better, 0 is perfectly fair)
        """
        U1 = utilities['U1']
        U2 = utilities['U2']
        
        ranks_s1 = get_preference_order(U1)
        ranks_s2 = get_preference_order(U2)
        
        sum_s1 = 0
        sum_s2 = 0
        
        for i in range(n_agents):
            match = matching.get(f's1_{i}')
            if match is not None:
                j = int(match.split('_')[1])
                sum_s1 += ranks_s1[i, j]
                sum_s2 += ranks_s2[j, i]
        
        return abs(sum_s1 - sum_s2)
    
    @staticmethod
    def compute_all_fairness_metrics(matching: Dict[str, str],
                                    utilities: Dict[str, np.ndarray],
                                    n_agents: int) -> Dict[str, float]:
        """
        Compute all fairness metrics at once.
        
        Returns:
            Dict with 'regret_cost', 'egalitarian_cost', 'set_equality_cost'
        """
        U1 = utilities['U1']
        U2 = utilities['U2']
        
        ranks_s1 = get_preference_order(U1)
        ranks_s2 = get_preference_order(U2)
        
        max_rank = 0
        total_cost = 0
        sum_s1 = 0
        sum_s2 = 0
        
        for i in range(n_agents):
            match = matching.get(f's1_{i}')
            if match is not None:
                j = int(match.split('_')[1])
                rank_i = ranks_s1[i, j]
                rank_j = ranks_s2[j, i]
                
                max_rank = max(max_rank, rank_i, rank_j)
                total_cost += rank_i + rank_j
                sum_s1 += rank_i
                sum_s2 += rank_j
        
        return {
            'regret_cost': max_rank,
            'egalitarian_cost': total_cost,
            'set_equality_cost': abs(sum_s1 - sum_s2)
        }
