import json

import numpy as np
import torch.random
import wandb

KEY = "INSERT KEY"


def common_setup_and_config(args):
    config_filename = args[1]
    wandb.login(key=KEY)
    config = None
    with open(config_filename, "r") as stream:
        try:
            config = json.load(stream)
        except json.JSONDecodeError as exc:
            print(exc)

    SEED = 7575
    torch.random.manual_seed(SEED)
    np.random.seed(SEED)
    return config
