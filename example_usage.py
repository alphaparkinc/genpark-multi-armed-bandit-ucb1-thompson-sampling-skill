"""Example usage for Multi-Armed Bandit Skill."""
from client import MultiArmedBandit

def main():
    print("Executing Multi-Armed Bandit (UCB1)...")
    mab = MultiArmedBandit(3)
    # True rewards: Arm 0 = 0.2, Arm 1 = 0.85, Arm 2 = 0.4
    for t in range(1, 200):
        arm = mab.select_arm_ucb1(t)
        reward = 1.0 if (arm == 1 and t % 5 != 0) else (1.0 if (arm == 2 and t % 3 == 0) else 0.0)
        mab.update(arm, reward)

    summary = mab.get_summary()
    print("MAB Summary:", summary)
    best_arm = max(range(3), key=lambda a: mab.counts[a])
    assert best_arm == 1, f"Expected Arm 1 to be pulled most, got {best_arm}"
    print("Multi-Armed Bandit verified successfully!")

if __name__ == "__main__":
    main()
