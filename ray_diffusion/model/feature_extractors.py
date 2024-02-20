import ipdb  # noqa: F401
import torch
import torch.nn as nn


def resize(image, size=None, scale_factor=None):
    return nn.functional.interpolate(
        image,
        size=size,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=False,
    )


class SpatialDino(nn.Module):
    def __init__(
        self,
        freeze_weights=True,
        model_type="dinov2_vits14",
        num_patches_x=16,
        num_patches_y=16,
    ):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_type)
        self.feature_dim = self.model.embed_dim
        self.num_patches_x = num_patches_x
        self.num_patches_y = num_patches_y
        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, autoresize=False):
        """
        Spatial dimensions of output will be H // 14, W // 14. If autoresize is True,
        then the output will be resized to the correct dimensions.

        Args:
            x (torch.Tensor): Images (B, C, H, W). Should be ImageNet normalized.
            autoresize (bool): Whether to resize the input to match the num_patch
                dimensions.

        Returns:
            feature_map (torch.tensor): (B, C, h, w)
        """
        *B, c, h, w = x.shape

        x = x.reshape(-1, c, h, w)

        # Output will be (B, H * W, C)
        features = self.model.forward_features(x)["x_norm_patchtokens"]
        features = features.permute(0, 2, 1)
        features = features.reshape(  # (B, C, H, W)
            -1, self.feature_dim, h // 14, w // 14
        )
        if autoresize:
            features = resize(features, size=(self.num_patches_y, self.num_patches_x))

        features = features.reshape(
            *B, self.feature_dim, self.num_patches_y, self.num_patches_x
        )
        return features
