import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt


#----Decode environment state----##
def decode_state(obs):
    taxi_row = obs // 25
    taxi_col = (obs % 25) // 5
    pass_loc = (obs % 25) % 5 // 5
    dest_idx = (obs % 25) % 5 % 5
    return taxi_row, taxi_col, pass_loc, dest_idx


##-------------------------------------##
def show_state(step, env, obs, reward):
    ansi_state = env.render()
    taxi_row, taxi_col, pass_loc, dest_idx = env.unwrapped.decode(obs)
    
    locations = ["R", "G", "Y", "B", "Taxi"]
    destinations = ["R", "G", "Y", "B"]
    
    pass_loc_meaning = locations[pass_loc]
    dest_idx_meaning = destinations[dest_idx]

    array_state = [taxi_row, taxi_col, pass_loc, dest_idx]
    print(f"Step {step}: {array_state} -> (Taxi is located in Row {taxi_row}, Column {taxi_col}. "
          f"Passenger location is {pass_loc_meaning} and destination is {dest_idx_meaning})\nReward: {reward}")
    print(ansi_state)


##-------------------------------------##
def initialize_environment(env_name="Taxi-v3"):
    env = gym.make(env_name, render_mode="ansi")
    obs, info = env.reset()
    print(f"The action space is: {env.action_space}")
    print(f"The state space is: {env.observation_space}")
    print("\nInitial environment state render:")
    show_state(0, env, obs, 0)
    return env


##-------------------------------------##
def train_agent(env, alpha, gamma, epsilon, n_episodes, max_steps):
   
   # Initialize Q-table
   q_table = np.zeros((env.observation_space.n, env.action_space.n)) 
   
   # Create lists to store episode and total reward
   episodes, rewards = [], []

   # Training
   for episode in range(n_episodes):
      
      obs, info = env.reset() # Reset environment
      total_reward = 0 # Initialize total reward

      # Agent randomly decided whether to "explore" or "exploit"
      for step in range(max_steps):
         if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
         else:
            action = np.argmax(q_table[obs]) # Exploit learned values

         next_obs, reward, done, truncated, info = env.step(action) # Take action
         total_reward += reward # Update total reward

         best_next_action = np.argmax(q_table[next_obs])
      
         #Update Q value
         q_table[obs, action] = q_table[obs, action] + alpha * (
            reward + gamma * q_table[next_obs, best_next_action] - q_table[obs, action]
         )
      
         obs = next_obs # Move to next state

         if done or truncated:
            break

      episodes.append(episode)
      rewards.append(total_reward)
      
      if episode % 100 == 0: # Print every 100 episodes
         print(f"Total reward {total_reward} for episode: {episode}")

   print("\nTraining has been completed.\n")
   return q_table, episodes, rewards


##-------------------------------------##
def test_agent(env, max_steps, q_table):
   # Ask user if they want to proceed with the test
   proceed_test = input("Do you want to proceed with the test? (Y/n): ")

   if proceed_test == "Y":
      # Test performance post-training
      obs, info = env.reset()
      show_state(0, env, obs, 0)

      total_reward = 0
      for step in range(max_steps):
         action = np.argmax(q_table[obs])
         obs, reward, terminated, truncated, info = env.step(action)
         show_state(step + 1, env, obs, reward)
         total_reward += reward

         if terminated or truncated:
            break

         print(f"Total reward during test: {total_reward}")
      else:
         print("Test will not be performed.")


##-------------------------------------##
def plot_results(episodes, rewards):
   # Ask user if they want to plot the results
   plot_results = input("Do you want to plot the training results? (Y/n): ")

   if plot_results == "Y":
      # Plot training 
      plt.plot(episodes, rewards)
      plt.xlabel('Episode')
      plt.ylabel('Total Reward')
      plt.title('Training Success')
      plt.show()
   else:
      print("Training results will not be plotted.")

# def plot_results(episodes, rewards, window=100):
#     # Ask user if they want to plot the results
#     plot_results = input("Do you want to plot the training results? (Y/n): ").lower()

#     if plot_results == "y":
#         # Calculate smoothed rewards
#         smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')

#         # Create subplots
#         fig, ax = plt.subplots(2, 1, figsize=(12, 8))

#         # Plot raw rewards
#         ax[0].plot(episodes, rewards, color='blue', alpha=0.3, label='Raw Rewards')
#         ax[0].plot(episodes[:len(smoothed_rewards)], smoothed_rewards, color='red', label=f'Smoothed Rewards ({window}-episode window)')
#         ax[0].set_xlabel('Episode')
#         ax[0].set_ylabel('Total Reward')
#         ax[0].set_title('Training Rewards')
#         ax[0].legend()

#         # Calculate and plot running average
#         running_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
#         ax[1].plot(episodes, running_avg, color='green', label='Running Average')
#         ax[1].set_xlabel('Episode')
#         ax[1].set_ylabel('Average Reward')
#         ax[1].set_title('Running Average Reward')
#         ax[1].legend()

#         # Adjust layout
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("Training results will not be plotted.")