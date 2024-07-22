# Imports
from src.functions import show_state
import numpy as np


##----Test agent----##
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

         print(f"Total reward: {total_reward}")
   else:
      print("Test will not be performed.")