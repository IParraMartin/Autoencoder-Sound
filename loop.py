import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassRecall, MulticlassPrecision
from torch.utils.data import DataLoader, TensorDataset

from tabulate import tabulate
import sys
import os
sys.path.append(os.path.curdir)


def get_table(metrics):
    headers = ['Metrics', 'Result']
    table = [[key, value] for key, value in metrics.items()]
    print(tabulate(table, headers, tablefmt="grid"))


def train(epochs: int, model: nn.Module, device: torch.device, train_dataloader: DataLoader, val_dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim, n_classes: int):

    print('Training...')
    model.to(device)
    print(f"Using device: {device}")

    all_T_train = []

    for epoch in range(epochs):
    
        model.train()
        train_loss = 0.0
        T_train = []
        t_accuracy = MulticlassAccuracy(num_classes=n_classes, average='micro').to(device)
        t_f1 = MulticlassF1Score(num_classes=n_classes, average='macro').to(device)
        t_recall = MulticlassRecall(num_classes=n_classes, average='macro').to(device)
        t_precision = MulticlassPrecision(num_classes=n_classes, average='macro').to(device)

        for idx_batch, (samples, targets) in enumerate(train_dataloader):
            samples, targets = samples.to(device), targets.to(device)

            preds, T = model(samples)
            loss = criterion(preds, targets)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # [B, n_C] -> [B, max(n_C) = 1]
            preds = torch.argmax(preds, dim=1)

            T_train.append(T)

            t_accuracy.update(preds, targets)
            t_f1.update(preds, targets)
            t_recall.update(preds, targets)
            t_precision.update(preds, targets)

            if (idx_batch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs} - Step [{idx_batch+1}/{len(train_dataloader)}] - Loss: {loss.item():.3f}]")
        
        train_loss /= len(train_dataloader)
    
        model.eval()
        val_loss = 0.0
        v_accuracy = MulticlassAccuracy(num_classes=n_classes, average='micro').to(device)
        v_f1 = MulticlassF1Score(num_classes=n_classes, average='macro').to(device)
        v_recall = MulticlassRecall(num_classes=n_classes, average='macro').to(device)
        v_precision = MulticlassPrecision(num_classes=n_classes, average='macro').to(device)

        with torch.no_grad():
            for idx_batch, (samples, targets) in enumerate(val_dataloader):
                samples, targets = samples.to(device), targets.to(device)
                preds, _ = model(samples)
                loss = criterion(preds, targets)
                val_loss += loss.item()

                preds = torch.argmax(preds, dim=1)

                v_accuracy.update(preds, targets)
                v_f1.update(preds, targets)
                v_recall.update(preds, targets)
                v_precision.update(preds, targets) 

                if (idx_batch + 1) % 10 == 0:
                    print(f"Validation Step [{idx_batch+1}/{len(val_dataloader)} - Loss: {loss.item():.3f}]")

        val_loss /= len(val_dataloader)

        print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.3f} - Val loss: {val_loss:.3f}")

        all_T_train.append(T_train)

    metrics = {
        'train_loss': train_loss,
        'train_acc': t_accuracy.compute(),
        'train_F1': t_f1.compute(),
        'train_recall': t_recall.compute(),
        'train_precision': t_precision.compute(),
        'val_loss': val_loss,
        'val_acc': v_accuracy.compute(),
        'val_F1': v_f1.compute(),
        'val_recall': v_recall.compute(),
        'val_precision': v_precision.compute()
    }

    get_table(metrics)
    
    return torch.stack([torch.stack(inner_list) for inner_list in all_T_train])


if __name__ == "__main__":

    def generate_fake_data(batch, channels, img_size, num_classes):
        X = torch.randn(batch, channels, img_size, img_size)
        y = torch.randint(0, num_classes, (batch,))
        return X, y


    img_size = 107
    channels = 1
    num_classes = 35
    n_train_samples = 6400
    n_val_samples = 1600
    batch_size = 128
    

    X_train, y_train = generate_fake_data(n_train_samples, channels, img_size, num_classes)
    X_val, y_val = generate_fake_data(n_val_samples, channels, img_size, num_classes)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)

    model = TrCNN(img_size=img_size, in_channels=channels, kernel_size=3, stride=2, padding=1,
                  n_classes=num_classes, t_dims=100, bias=False, T_dims=300)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    T_space = train(epochs=20, model=model, device=device, train_dataloader=train_dataloader, 
              val_dataloader=val_dataloader, criterion=criterion, optimizer=optimizer, n_classes=num_classes)
    
    # print(type(T_space))
    # print(T_space.shape)

    # def save_space(latent_space):
    #     os.makedirs('spaces', exist_ok=True)
    #     torch.save(latent_space, 'spaces/beta_space.pt')

    # save_space(T_space)
    # Out:
    # [Epochs, Batch, Last Layer Out CNN, T Dim] -> [E, B, LL, T_D]