# Imports
from src.functions import initialize_environment
from src.training import train_agent
from src.testing import test_agent


def main():
    # Environment initialization
    env = initialize_environment()


    # Hyperparameters
    alpha = 0.1    # Learning rate
    gamma = 0.99   # Discount factor
    epsilon = 0.1  # Exploration rate
    n_episodes = 10000  # Number of episodes
    max_steps = 200   # Maximum steps per episode
    done = False # Termination flag


    # Train agent
    q_table = train_agent(env, alpha, gamma, epsilon, n_episodes, max_steps)


    # Test agent
    test_agent(env, max_steps, q_table)


if __name__ == "__main__":
    main()