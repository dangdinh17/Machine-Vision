import torch
import os
def save_checkpoint(model, optimizer, scheduler, epoch, train_step, test_step, best_loss, path="checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'train_step': train_step,
        'test_step':test_step,
        'best_loss': best_loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, scheduler, path="checkpoint.pth"):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        train_step = checkpoint['train_step']
        val_step = checkpoint['test_step']
        best_loss = checkpoint['best_loss']
        print(f"Checkpoint loaded from {path}")
    else:
        print(f"No checkpoint found at {path}")
        epoch, train_step, val_step, best_loss = 0, 0, 0, float('inf')
    return epoch, train_step, val_step, best_loss