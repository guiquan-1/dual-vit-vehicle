import torch
import torch.nn as nn
import timm  # Install with: pip install timm

class DualStreamViT(nn.Module):
    def __init__(self, vit_model='vit_base_patch16_224', fusion='concat', num_classes=2):
        super().__init__()
        self.rgb_vit = timm.create_model(vit_model, pretrained=True)
        self.thermal_vit = timm.create_model(vit_model, pretrained=True)

        self.rgb_vit.reset_classifier(0)
        self.thermal_vit.reset_classifier(0)

        hidden_dim = self.rgb_vit.embed_dim  # 768 for base ViT

        self.fusion = fusion
        fusion_dim = hidden_dim * 2 if fusion == 'concat' else hidden_dim

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb, thermal):
        rgb_tokens = self.rgb_vit.forward_features(rgb)      # [B, N, D]
        thermal_tokens = self.thermal_vit.forward_features(thermal)  # [B, N, D]

        if self.fusion == 'concat':
            fused = torch.cat((rgb_tokens[:, 0], thermal_tokens[:, 0]), dim=-1)  # Use [CLS] tokens
        elif self.fusion == 'add':
            fused = rgb_tokens[:, 0] + thermal_tokens[:, 0]
        else:
            raise ValueError("Fusion type not supported")

        return self.head(fused)
