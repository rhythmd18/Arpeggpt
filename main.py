import torch
from arpeggpt.config import get_config
from arpeggpt.train import main

if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    GPT_CONFIG = get_config('test')
    # print(DEVICE)
    # print(GPT_CONFIG)
    main(
        batch_size=16, 
        num_epochs=20, 
        save_every=4, 
        config=GPT_CONFIG, 
        device=DEVICE
    )