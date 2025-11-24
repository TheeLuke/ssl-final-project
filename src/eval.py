import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os

from data import get_dataloaders
from model import get_model


#k-NN Implementation
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # feature: [1, D]
    # feature_bank: [N, D]
    # feature_labels: [N]
    
    # Compute Cosine Similarity
    # (We assume features are already normalized)
    sim_matrix = torch.mm(feature, feature_bank.T) # [1, N]
    
    # Get top K neighbors
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1) # [1, K]
    
    # Weighted Voting
    sim_labels = torch.gather(feature_labels.expand(feature.shape[0], -1), dim=-1, index=sim_indices)
    
    # Temperature scaling for voting weight
    sim_weight = (sim_weight / knn_t).exp()
    
    # Count votes
    one_hot_label = torch.zeros(feature.shape[0] * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), src=sim_weight.view(-1, 1))
    
    # Sum up votes
    pred_scores = torch.sum(one_hot_label.view(feature.shape[0], -1, classes), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    
    return pred_labels


#Feature Extraction Loop
@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    
    for img, label in tqdm(loader, desc="Extracting Features"):
        img = img.to(device)
        # We only need 'h' (the backbone output), not 'z' (projection)
        # Check model.py return signature: return h, z
        h, _ = model(img)
        
        # Normalize for Cosine Similarity
        h = F.normalize(h, dim=1)
        
        features.append(h)
        labels.append(label)
        
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0).to(device)
    return features, labels

def evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # 1. oad Data
    _, memory_loader, test_loader = get_dataloaders(
        dataset_name=args.dataset, 
        batch_size=args.batch_size
    )

    #Load Model
    model = get_model(args.arch).to(device)
    
    #Load Checkpoint
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Handle state_dict keys (remove 'module.' if saved with DataParallel)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"No checkpoint found at {args.checkpoint}")

    #Extract Features
    print("Building Feature Bank (Train Set)...")
    memory_features, memory_labels = extract_features(model, memory_loader, device)
    
    print("Extracting Test Features...")
    test_features, test_labels = extract_features(model, test_loader, device)
    
    print("Running k-NN Inference...")
    # Adjust classes based on dataset (CIFAR10=10)
    num_classes = 10 
    
    total_top1 = 0
    total_num = 0
    
    # Process test images in chunks to save memory
    test_bar = tqdm(range(test_features.shape[0]), desc="k-NN")
    for i in test_bar:
        # Get single test feature
        feat = test_features[i].unsqueeze(0) # [1, D]
        target = test_labels[i]
        
        # Predict
        pred_labels = knn_predict(feat, memory_features, memory_labels, num_classes, args.k, args.t)
        
        # Check Accuracy
        total_num += 1
        if pred_labels[0, 0] == target:
            total_top1 += 1
            
        test_bar.set_postfix(Acc=f"{(total_top1/total_num)*100:.2f}%")

    print(f"Final k-NN Accuracy: {(total_top1/total_num)*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth file")
    parser.add_argument("--k", type=int, default=200, help="k in k-NN")
    parser.add_argument("--t", type=float, default=0.1, help="temperature in k-NN")
    
    args = parser.parse_args()
    evaluation(args)