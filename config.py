seed = 42 # the random seed 
env = "PongNoFrameskip-v4" # the name of the game
buffer_size = int(5e3) # the size of the memory buffer
learning_rate = 1e-4 # learning rate
discount_rate = 0.99 # discount rate
num_steps = int(2e6) # the total steps will be played
batch_size = 32 # batch size
learning_start = int(1e4) # the number of steps before learning begin
learning_freq = 1 # the number of steps between every two optimization steps
target_update_freq = int(1e3) # the number of steps between every two target network update 
eps_start = 1 # e-greedy start threshold
eps_end = 0.01 # e-greedy end threshold
eps_fra = 0.1 # fraction of num steps
print_freq = 10 # the frequence of print
gpu_id = 6 # the id of gpu used

video_path = "./video"