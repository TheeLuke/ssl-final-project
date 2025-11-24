import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import argparse
import os
from tqdm import tqdm

# Import our custom modules
from data import get_dataloaders
from model import get_model


# The SimCLR Loss Function (NT-Xent)
class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        
        # Concatenate both views: [2*B, Dim]
        z = torch.cat([z_i, z_j], dim=0)
        
        # Normalize the vectors (critical for contrastive learning)
        z = F.normalize(z, dim=1)
        
        # Compute similarity matrix (Cosine Similarity)
        # sim[a, b] = z[a] dot z[b]
        similarity_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Create labels: The positive pair for i is (i + batch_size)
        # maximize diagonal offsets
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ], dim=0).to(z.device)
        
        # Mask out self-similarity (similarity of image with itself)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
        # Compute Cross Entropy
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


#Training Loop
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    #Load Data
    # Use 'project_data' for the 500k images, 'cifar10' for sanity check
    train_loader, _, _ = get_dataloaders(
        dataset_name=args.dataset, 
        batch_size=args.batch_size, 
        num_workers=4
    )
    
    #Initialize Model
    model = get_model(args.arch).to(device)
    
    #Setup Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = NTXentLoss(temperature=args.temperature)
    
    # Scheduler: Cosine Decay is standard for SSL
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    
    #Mixed Precision Scaler
    scaler = GradScaler()
    
    print(f"Starting training for {args.epochs} epochs...")
    model.train()
    
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for batch_idx, (images, _) in enumerate(progress_bar):
            # images is a list: [view1, view2]
            x1, x2 = images[0].to(device), images[1].to(device)
            
            optimizer.zero_grad()
            
            # Autocast for Mixed Precision (Faster)
            with autocast():
                # Forward pass
                # We only need the projection 'z' for the loss
                _, z1 = model(x1)
                _, z2 = model(x2)
                
                loss = criterion(z1, z2)
            
            # Backward pass with Scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save every 5 epochs OR if it's the last epoch
        if epoch % 5 == 0 or epoch == args.epochs:
            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/ssl_{args.arch}_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Checkpoint saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSL Training Script")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "project_data"], help="Dataset to use")
    parser.add_argument("--arch", type=str, default="resnet18", help="Model backbone architecture")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for NT-Xent loss")
    
    args = parser.parse_args()
    train(args)