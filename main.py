import torch
from arpeggpt.config import get_config
from arpeggpt.train import main

if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    GPT_CONFIG = get_config('small')
    # print(DEVICE)
    # print(GPT_CONFIG)
    main(16, 20, 4, GPT_CONFIG, DEVICE)