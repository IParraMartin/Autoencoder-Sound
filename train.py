import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

import os

from tabulate import tabulate
from loss.loss import vae_loss_function


def get_table(metrics):
    headers = ['Metric', 'Result']
    table = [[key, value] for key, value in metrics.items()]
    print(tabulate(table, headers, tablefmt='grid'))


def save_checkpoint(model, epoch):
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f'checkpoints/vae_checkpoint_epoch{epoch}.pt')


def train(epochs: int, model: nn.Module, device: torch.device, train_dataloader: DataLoader, val_dataloader: DataLoader, optimizer: Optimizer, save_model: bool):

    model = model.to(device)
    print(f'Device: {device}')
    print('Training...')

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        for idx_batch, samples in enumerate(train_dataloader):
            samples = samples.to(device)

            outputs, mu, log_var = model(samples)
            loss = vae_loss_function(outputs, mu, log_var)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (idx_batch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}] - Step [{idx_batch+1}/{len(train_dataloader)}] - loss: {loss.item():.3f}')

        train_loss /= len(train_dataloader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for idx_batch, samples in enumerate(val_dataloader):
                samples = samples.to(device)

                outputs, mu, log_var = model(samples)
                loss = vae_loss_function(outputs, mu, log_var)
                val_loss += loss.item()

                if (idx_batch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}] - Step [{idx_batch+1}/{len(val_dataloader)}] - loss: {loss.item():.3f}')

            val_loss /= len(val_dataloader)

        if save_model and (epoch + 1) % 5 == 0:
            save_checkpoint(model, epoch+1)
            print('Checkpoint generated.')

    metrics = {
        'final_train_loss': train_loss,
        'final_val_loss': val_loss
    }
    
    get_table(metrics)
