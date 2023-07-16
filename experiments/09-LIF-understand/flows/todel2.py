# %%
# Import the W&B Python Library and log into W&B
import wandb
wandb.login()

# 1: Define objective/training function
def objective(config):
    score = config.x ** 3 + config.y
    return score

def main():
    wandb.init(project='my-first-sweep')
    score = objective(wandb.config)
    wandb.log({'score': score})




# 2: Define the search space
sweep_configuration = {
    'method': 'random',
    'metric': 
    {
        'goal': 'minimize', 
        'name': 'score'
        },
    'parameters': 
    {
        'x': [1, 2, 3],
        'y': [4, 5],
     }
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project='my-first-sweep'
    )

wandb.agent(sweep_id, function=main, count=10)
# %%
