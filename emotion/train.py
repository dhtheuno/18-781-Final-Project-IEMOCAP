import torch
import torch.nn.functional as F
import time
import argparse
import numpy as np
from collections import Counter
from tqdm import tqdm
import os
import random
from torch.utils.tensorboard import SummaryWriter

from configs import get_config

def train(config, save_path, tb_path):
    # redefine config variables for readability
    model = config.model
    optimizer = config.optimizer
    scheduler = config.scheduler
    loss_fn = config.loss_fn
    train_loader = config.train_loader
    val_loader = config.val_loader
    n_epochs = config.n_epochs

    writer = SummaryWriter(tb_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    since = time.time()
    best_val_loss = np.inf

    for epoch in range(n_epochs):
        print(f'Epoch {epoch}/{n_epochs - 1}')
        print('-' * 10)

        # train epoch
        model.train()
        running_loss = 0.0
        running_corrects = 0
        pred_counter = Counter()

        for inputs, labels in tqdm(train_loader):

            # train current batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # update statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            pred_counter.update(preds.data.cpu().numpy())

        scheduler.step()

        # print statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * running_corrects.double() / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Prediction Counts: {dict(pred_counter)}')
        val_loss, val_acc = test(model, config.val_loader, split='val')

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f'saving model')
            torch.save(model.state_dict(), save_path)

        # tensorboard logging
        writer.add_scalar('loss/train', epoch_loss, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('loss/val_best', best_val_loss, epoch)
        writer.add_scalar('accuracy/train', epoch_acc, epoch)
        writer.add_scalar('accuracy/val', val_acc, epoch)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    writer.flush()
    writer.close()
    return model

def test(model, test_loader, split='Test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_counter = Counter()
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            loss += F.cross_entropy(output, labels).item() * inputs.size(0)
            preds = torch.max(output.data, 1)[1]
            pred_counter.update(preds.data.cpu().numpy())
            correct += (preds == labels).sum()
    loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'{split} Loss: {loss}, {split} Acc: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%), Prediction Counts: {dict(pred_counter)}')
    return loss, accuracy

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Train a model given a config tag')
    parser.add_argument('--tag', required=True, type=str, help='configuration tag, see config.py for options')
    parser.add_argument('--run', required=False, type=str, default='', help='suffix for tensorboard run')
    parser.add_argument('--seed', required=False, type=int, default=0, help='manual seed for torch/numpy/random')
    parser.add_argument('--model-dir', required=False, type=str, default='saved_models/', help='folder to save model')
    args = parser.parse_args()

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # get config and run model
    config = get_config(args.tag)
    os.makedirs(args.model_dir, exist_ok=True)
    run_name = f'{args.tag}_{args.run}' if args.run else args.tag
    save_path = os.path.join(args.model_dir, f'{run_name}.pth')
    tb_path = os.path.join('./tensorboard/', f'{run_name}')

    model = train(config, save_path, tb_path)

    # test final model
    print('Last Model:')
    test(model, config.test_loader)

    # test best model (early stopping)
    print('Best Model:')
    model.load_state_dict(torch.load(save_path))
    test(model, config.test_loader)
