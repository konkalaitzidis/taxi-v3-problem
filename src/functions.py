##----Imports----##
import gymnasium as gym


##----Decode environment state----##
def decode_state(obs):
    taxi_row = obs // 25
    taxi_col = (obs % 25) // 5
    pass_loc = (obs % 25) % 5 // 5
    dest_idx = (obs % 25) % 5 % 5
    
    return taxi_row, taxi_col, pass_loc, dest_idx


##----Show environment state----##
def show_state(step, env, obs, reward):
    ansi_state = env.render()
    taxi_row, taxi_col, pass_loc, dest_idx = env.unwrapped.decode(obs)
    
    locations = ["R", "G", "Y", "B", "Taxi"]
    destinations = ["R", "G", "Y", "B"]
    
    pass_loc_meaning = locations[pass_loc]
    dest_idx_meaning = destinations[dest_idx]

    array_state = [taxi_row, taxi_col, pass_loc, dest_idx]

    print(f"Step {step}: {array_state} -> (Taxi is located in Row {taxi_row} - Column {taxi_col}. The passenger's location is {pass_loc_meaning} and destination is {dest_idx_meaning})\nReward: {reward}")
    print(ansi_state)


##----Initialize environment----##
def initialize_environment(env_name="Taxi-v3"):
    env = gym.make(env_name, render_mode="ansi")
    obs, info = env.reset()

    print(f"The action space is: {env.action_space}")
    print(f"The state space is: {env.observation_space}")
    print("\nInitial environment state render:")

    show_state(0, env, obs, 0)

    return env