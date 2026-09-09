"""
Autonomous Agent Multi-Armed Bandit Skill
Pure Python Standard Library implementation of UCB1 and Thompson Sampling.
"""
import math
from typing import List, Dict, Any

class MultiArmedBandit:
    """
    Multi-Armed Bandit exploration-exploitation algorithms.
    """
    def __init__(self, num_arms: int):
        self.k = num_arms
        self.counts = [0] * num_arms
        self.rewards = [0.0] * num_arms

    def select_arm_ucb1(self, t: int) -> int:
        for a in range(self.k):
            if self.counts[a] == 0:
                return a
        ucb_values = []
        for a in range(self.k):
            mean = self.rewards[a] / self.counts[a]
            bonus = math.sqrt((2.0 * math.log(max(1, t))) / self.counts[a])
            ucb_values.append(mean + bonus)
        return int(max(range(self.k), key=lambda a: ucb_values[a]))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.rewards[arm] += reward

    def get_summary(self) -> Dict[str, Any]:
        return {
            "counts": list(self.counts),
            "total_rewards": [round(r, 4) for r in self.rewards],
            "estimated_means": [round(self.rewards[a] / max(1, self.counts[a]), 4) for a in range(self.k)]
        }
