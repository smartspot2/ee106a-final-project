import wandb


def init_wandb(args):
    wandb.init(project="eecs106a-final-project", config=vars(args))


def log_scalar(step, key, scalar):
    wandb.log({key: scalar}, step=step)
