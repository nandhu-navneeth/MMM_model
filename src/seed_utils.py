import os, random, numpy as np

def set_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(seed)
    np.random.seed(seed)

