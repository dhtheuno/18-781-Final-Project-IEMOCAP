import argparse
import torch
import torch.nn.functional as F
import os
from collections import Counter
from tqdm import tqdm

from datasets import get_loader
from configs import get_config
from train import test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get test set results for a saved model')
    parser.add_argument('--model-dir', required=False, type=str, default='saved_models/', help='folder to save model')
    parser.add_argument('--tag', required=True, help='filename of saved model')
    parser.add_argument('--run', required=True, help='model suffix')
    args = parser.parse_args()

    testloader = get_loader(split='test')
    model = get_config(args.tag).model
    run_name = f'{args.tag}_{args.run}.pth' if args.run else f'{args.tag}.pth'
    model.load_state_dict(torch.load(os.path.join(args.model_dir, run_name)))

    test(model, testloader)





