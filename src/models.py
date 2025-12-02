import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# We rely on timm for the robust ViT implementation
# pip install timm
import timm
from timm.models.vision_transformer import VisionTransformer

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        """
        The projection head used in DINO.
        It projects the backbone features (384 dim) into a higher dim space
        where the clustering/prototype matching happens.
        """
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
        
        # The last layer is the "Prototype" layer.
        # It maps the bottleneck features to the number of "classes" (out_dim).
        # In DINO, out_dim is usually large (e.g. 65536) to allow for many clusters.
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        
        # Fix the magnitude of the last layer to 1. This stabilizes training.
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class MultiCropWrapper(nn.Module):
    """
    A wrapper that handles the "list of crops" logic.
    Instead of passing 1 tensor, the dataloader returns a list of tensors 
    (2 global + 6 local). This wrapper concatenates them for efficiency.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # x is a list of tensors (e.g., [global_1, global_2, local_1, ...])
        
        # Optimization: We want to run the backbone on all crops in parallel 
        # if they have the same resolution.
        # Since we resized ALL crops to 96x96 in dataset.py, we can 
        # concatenate them all into one massive batch.
        
        if isinstance(x, list):
            # Concatenate along batch dimension
            # Shape becomes: (Batch_Size * Num_Crops, 3, 96, 96)
            idx_crops = torch.cumsum(torch.unique_tensor(
                torch.tensor([inp.shape[0] for inp in x])
            ), 0)
            
            start_idx = 0
            for end_idx in idx_crops:
                _ = self.backbone(torch.cat(x[start_idx: end_idx]))
                start_idx = end_idx
            
            # Run backbone on the massive batch
            combined_input = torch.cat(x)
            
            # Forward pass through ViT
            # ViT output is (N, Dim) - usually the CLS token
            features = self.backbone(combined_input)
            
            # Forward pass through DINO head
            return self.head(features)
        else:
            # Single tensor case (used during evaluation)
            features = self.backbone(x)
            return self.head(features)

def get_vit_small_dino(patch_size=8, img_size=96, out_dim=65536):
    """
    Factory function to create the Student or Teacher model.
    """
    # 1. Instantiate ViT-Small
    # We use 'vit_small_patch16_224' as a base but override critical params
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,  # CRITICAL: Set to 8 for 96px images
        embed_dim=384,          # ViT-Small dimension
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    
    # 2. Instantiate DINO Head
    # Embed dim must match backbone (384 for ViT-S)
    head = DINOHead(
        in_dim=384,
        out_dim=out_dim,
        bottleneck_dim=256
    )
    
    # 3. Wrap them
    wrapper = MultiCropWrapper(backbone=model, head=head)
    
    return wrapper

if __name__ == "__main__":
    # Sanity Check
    print("Testing ViT-S/8 model definition...")
    model = get_vit_small_dino(patch_size=8, img_size=96)
    
    # Simulate a batch of 2 global crops and 6 local crops
    # Batch size 4
    dummy_input = [torch.randn(4, 3, 96, 96) for _ in range(8)]
    
    output = model(dummy_input)
    print(f"Output shape (Batch*Crops, Out_Dim): {output.shape}")
    
    # Expected: (4 * 8, 65536) = (32, 65536)
    assert output.shape == (32, 65536)
    print("Sanity check passed!")