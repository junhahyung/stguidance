# ----------- Mode Options -----------
mode = "STG-R"  # Choose from ["CFG", "STG-A", "STG-R"]

# ----------- STG Options -----------
stg_block_idx = [34, 35]  # List of block indices e.g., [35]
stg_scale = 1.0  # (e.g., 1.0)

# ----------- Rescaling Options -----------
do_rescaling = True  
rescaling_scale = 0.7 

# ----------- Restart Options -----------
do_restart = False  
restart_idx = 63  
num_restarts = 1  
max_idx = 1 
num_intervals = -1 

# ----------- CFG Scale and Steps -----------
cfg_scale = 4.5  # The scale value for the CFG (e.g., 4.5 is a moderate strength)
num_steps = 64  # Number of steps for model inference (e.g., 64 steps)

# ----------- Prompt Options -----------
prompt = None  # A single prompt for video generation / None for no prompt
prompts_path = "prompts/demo_prompts2.txt"  # Path to the prompts file (if used instead of a single prompt) / None for no prompts file

# ----------- Video Options -----------
width = 800  
height = 480 
num_frames = 91 
seed = 1710977262 

# ----------- Model Directory -----------
model_dir = "./weights"  # The directory path to the model

# ----------- CPU Offload -----------
cpu_offload = False  # Whether to offload model to CPU