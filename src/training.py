# Imports
import logging
import random
import numpy as np


##----Train agent----##
def train_agent(env, alpha, gamma, epsilon, n_episodes, max_steps):
   print("\nTraining in progress :) ")
   
   # Initialize Q-table
   q_table = np.zeros((env.observation_space.n, env.action_space.n)) 


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

      
      if episode % 1000 == 0: # Print every 1000 episodes
         print(f"Total reward {total_reward} for episode: {episode}")


   print("\nTraining has been completed.\n")
   return q_table