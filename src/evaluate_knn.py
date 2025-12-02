import argparse
import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import utils
from models import get_vit_small_dino

def extract_features(model, loader, use_cuda=True):
    """
    Extracts features from a dataloader using the given model.
    Returns:
        features: (N, Dim) tensor
        labels: (N,) tensor
    """
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for samples, targets in tqdm(loader, desc="Extracting features"):
            if use_cuda:
                samples = samples.cuda(non_blocking=True)
            
            # Forward pass through backbone only (not the head)
            # The model wrapper in models.py returns head(backbone(x))
            # We need to bypass the head.
            # Access the backbone directly:
            output = model.backbone(samples)
            
            # Handle different timm versions (some return sequence, some CLS)
            if output.ndim == 3:
                output = output[:, 0] # Take CLS token
            
            # L2 Normalize the features (Crucial for KNN!)
            output = nn.functional.normalize(output, dim=1, p=2)
            
            features.append(output.cpu())
            labels.append(targets)
            
    return torch.cat(features), torch.cat(labels)

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    """
    Weighted k-NN classifier.
    """
    top1 = 0
    total = 0
    train_features = train_features.cuda()
    train_labels = train_labels.cuda()
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()

    num_test_images = test_labels.shape[0]
    chunk_size = 1000 # Process in chunks to save memory
    
    for idx in range(0, num_test_images, chunk_size):
        # get the features for this batch
        features_batch = test_features[idx : min((idx + chunk_size), num_test_images)]
        targets_batch = test_labels[idx : min((idx + chunk_size), num_test_images)]
        batch_size = targets_batch.shape[0]

        # Calculate Cosine Similarity (since features are normalized, dot product = cosine sim)
        # (Batch, Dim) @ (Dim, Train_Size) -> (Batch, Train_Size)
        sim_matrix = torch.mm(features_batch, train_features.t())

        # Get top K neighbors
        sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)

        # Weighted voting: e^(sim / T)
        sim_weight = (sim_weight / T).exp()

        # Gather the labels of the neighbors
        # (Batch, k)
        one_hot_label = torch.zeros(batch_size * k, num_classes, device=sim_indices.device)
        neighbor_labels = train_labels.view(1, -1).expand(batch_size, -1)
        # Select the labels corresponding to the top k indices
        neighbor_labels = torch.gather(neighbor_labels, 1, sim_indices)
        
        # Weighted Voting
        one_hot_label = one_hot_label.scatter(1, neighbor_labels.view(-1, 1), 1)
        pred_scores = torch.sum(one_hot_label.view(batch_size, k, num_classes) * sim_weight.unsqueeze(2), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        
        # Check Top-1 accuracy
        top1 += torch.sum((pred_labels[:, :1] == targets_batch.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total += batch_size

    return top1 / total * 100

def eval_dataset(name, train_path, val_path, model, k, T):
    print(f"\nEvaluating on {name}...")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"Skipping {name} (Data not found at {train_path})")
        return

    # Standard Transform (No Augmentation, just Resize)
    # MUST match training resolution (96x96)
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_ds = datasets.ImageFolder(train_path, transform=transform)
    val_ds = datasets.ImageFolder(val_path, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    print(f"  Extracting Support Features ({len(train_ds)} images)...")
    train_features, train_labels = extract_features(model, train_loader)
    
    print(f"  Extracting Query Features ({len(val_ds)} images)...")
    val_features, val_labels = extract_features(model, val_loader)

    print("  Running k-NN...")
    # Determine number of classes dynamically
    num_classes = len(train_ds.classes)
    acc = knn_classifier(train_features, train_labels, val_features, val_labels, k, T, num_classes)
    
    print(f"  {name} Top-1 Accuracy: {acc:.2f}%")
    return acc

def main():
    parser = argparse.ArgumentParser('DINO k-NN Evaluation')
    parser.add_argument('--checkpoint_path', default='./output_dir/checkpoint.pth', type=str)
    parser.add_argument('--patch_size', default=8, type=int)
    parser.add_argument('--nb_knn', default=20, type=int, help='Number of NN to use')
    parser.add_argument('--temperature', default=0.07, type=float, help='Temperature for KNN')
    
    # Paths to the LABELED data generated by prepare_*.py scripts
    parser.add_argument('--cub_dir', default='./kaggle_data') 
    parser.add_argument('--sun_dir', default='./kaggle_data_sun397')
    parser.add_argument('--mini_dir', default='./kaggle_data_miniimagenet')
    
    args = parser.parse_args()

    # 1. Load Model
    print(f"Loading model from {args.checkpoint_path}")
    # Note: We don't care about the DINO Head here, just the backbone.
    # But we load the full wrapper to match keys.
    model_wrapper = get_vit_small_dino(patch_size=args.patch_size)
    model_wrapper.cuda()
    
    # Load weights
    if os.path.isfile(args.checkpoint_path):
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        # Load Teacher weights (usually better performance than student)
        if "teacher" in state_dict:
            msg = model_wrapper.load_state_dict(state_dict["teacher"], strict=False)
            print(f"Loaded Teacher weights: {msg}")
        else:
            print("Error: Checkpoint does not contain 'teacher' key.")
            return
    else:
        print(f"Error: No checkpoint found at {args.checkpoint_path}")
        return

    # 2. Evaluate on All 3 Domains
    # CUB-200 (Birds)
    eval_dataset("CUB-200 (Birds)", 
                 os.path.join(args.cub_dir, 'train'), 
                 os.path.join(args.cub_dir, 'val'), 
                 model_wrapper, args.nb_knn, args.temperature)

    # SUN397 (Scenes)
    eval_dataset("SUN397 (Scenes)", 
                 os.path.join(args.sun_dir, 'train'), 
                 os.path.join(args.sun_dir, 'val'), 
                 model_wrapper, args.nb_knn, args.temperature)

    # Mini-ImageNet (Objects)
    eval_dataset("Mini-ImageNet (Objects)", 
                 os.path.join(args.mini_dir, 'train'), 
                 os.path.join(args.mini_dir, 'val'), 
                 model_wrapper, args.nb_knn, args.temperature)

if __name__ == '__main__':
    main()