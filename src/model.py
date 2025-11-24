import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

class SSLResNet(nn.Module):
    def __init__(self, arch='resnet18', dim=128, hidden_dim=2048):
        super().__init__()
        
        #Backbone/Base Encoder
        #standard ResNet but drop the final classification layer
        if arch == 'resnet18':
            backbone = resnet18()
            prev_dim = 512
        elif arch == 'resnet50':
            backbone = resnet50()
            prev_dim = 2048
        
        # Remove the last fully connected layer to get the features
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        #Projection Head, SSL
        # Standard SimCLR/MoCo head: Linear -> ReLU -> Linear
        # This maps the features to a lower dimensional space for the loss function
        self.projection_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        # x shape: [batch, 3, 96, 96]
        
        # Extract features
        h = self.backbone(x) # [batch, prev_dim, 1, 1]
        h = h.flatten(start_dim=1) # [batch, prev_dim]
        
        # Project to embedding space
        z = self.projection_head(h) # [batch, dim]
        
        return h, z

def get_model(name='resnet18'):
    model = SSLResNet(arch=name)
    return model

if __name__ == "__main__":
    model = get_model('resnet18')
    
    # Paramter count must be < 100M. safety check
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    if total_params < 100_000_000:
        print("SUCCESS: Model is under the 100M parameter limit.")
    else:
        print("FAIL: Model is too large!")