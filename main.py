# Imports
from src.functions import show_state
from src.functions import initialize_environment
from src.functions import train_agent
from src.functions import plot_results
from src.functions import test_agent


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
    q_table, episodes, rewards = train_agent(env, alpha, gamma, epsilon, n_episodes, max_steps)


    # Test agent
    test_agent(env, max_steps, q_table)


    #TODO add other things like analysis??


if __name__ == "__main__":
    main()