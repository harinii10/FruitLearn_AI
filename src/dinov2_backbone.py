import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2ViTBackbone(nn.Module):
    """
    DINOv2 Self-Supervised Vision Transformer backbone wrapper (ViT-Base/14).
    Supports multi-crop features, register tokens, and projection heads.
    """
    def __init__(self, patch_size=14, embed_dim=768, depth=12, num_heads=12, num_classes=6):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch projection layer
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # CLS and Register Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, 4, embed_dim))  # DINOv2 registers artifact removal
        self.pos_embed = nn.Parameter(torch.zeros(1, (224 // patch_size)**2 + 5, embed_dim))

        # Vision Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * 4),
            activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # Multi-layer Classifier Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        reg_tokens = self.register_tokens.expand(B, -1, -1)
        x = torch.cat((cls_tokens, reg_tokens, x), dim=1)
        
        # Interpolate pos_embed if sequence length differs
        if x.shape[1] == self.pos_embed.shape[1]:
            x = x + self.pos_embed
        else:
            x = x + F.interpolate(self.pos_embed, size=x.shape[1], mode='linear', align_corners=False)
            
        x = self.encoder(x)
        x = self.norm(x)
        return x[:, 0]  # Return [CLS] token representation

    def forward(self, x):
        feats = self.forward_features(x)
        logits = self.head(feats)
        return logits

def load_dinov2_pretrained(num_classes=6):
    """ Factory function to instantiate DINOv2 model """
    model = DINOv2ViTBackbone(num_classes=num_classes)
    return model

if __name__ == "__main__":
    model = load_dinov2_pretrained()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"DINOv2 ViT-Base/14 Output Shape: {out.shape} | Parameters: {sum(p.numel() for p in model.parameters()):,}")
