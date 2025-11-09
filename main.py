import torch
import argparse
from arpeggpt.config import get_config
from arpeggpt.train import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='test', help='The GPT config to use.')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size.')
    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to train for.')
    parser.add_argument('--save_every', type=int, default=4, help='Save the model every n epochs.')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    GPT_CONFIG = get_config(args.config)
    main(
        batch_size=args.batch_size, 
        num_epochs=args.num_epochs, 
        save_every=args.save_every, 
        config=GPT_CONFIG, 
        device=DEVICE
    )