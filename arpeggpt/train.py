import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from arpeggpt.data import prepare_dataset
from arpeggpt.model import GPTModel

def calculate_loss_batch(input_batch, target_batch, attn_mask, model):
    '''
    Calculate the cross-entropy loss for a batch of data.
    Args:
        input_batch: The input tensor batch.
        target_batch: The target tensor batch.
        attn_mask: The attention mask tensor.
        model: The GPT model.
    Returns:
        loss: The computed cross-entropy loss.
    '''
    logits = model(input_batch, attn_mask)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return loss

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        save_every: int,
        device: torch.device
    ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_every = save_every
        self.device = device
        self.model.to(self.device)

    def _run_batch(self, X, y, attn_mask):
        self.optimizer.zero_grad()
        loss = calculate_loss_batch(X, y, attn_mask, self.model)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step(loss)
        return loss

    def _run_epoch(self):
        total_loss = 0
        num_batches = len(self.dataloader)
        for i, batch in enumerate(self.dataloader):
            X, y, attn_mask = (
                batch['input_ids'].to(self.device), 
                batch['labels'].to(self.device), 
                batch['attention_mask'].to(self.device)
            )
            batch_loss = self._run_batch(X, y, attn_mask)
            total_loss += batch_loss
            if i % 100 == 0:
                print(f'    Batch: {i+1}, Loss: {batch_loss:.4f}')
        avg_loss = total_loss / num_batches
        print(f'Average Loss: {avg_loss}\n')

    def _save_checkpoint(self, epoch):
        checkpoint = self.model.state_dict()
        PATH = f'model_checkpoints/checkpoint{epoch}.pt'
        torch.save(checkpoint, PATH)
        print(f"Training checkpoint saved at {PATH}\n")

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch + 1}\n----------------------------------------')
            self._run_epoch()
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        torch.save(self.model, 'arpeggpt.pth')


def load_train_objects(batch_size, config):
    dataloader = prepare_dataset(batch_size, config)
    model = GPTModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)
    return dataloader, model, optimizer, lr_scheduler


def main(batch_size: int, num_epochs: int, save_every: int, config: dict, device: torch.device):
    dataloader, model, optimizer, lr_scheduler = load_train_objects(batch_size, config)
    trainer = Trainer(
        model=model, 
        dataloader=dataloader, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler, 
        save_every=save_every, 
        device=device
    )
    model.train()
    trainer.train(num_epochs)
