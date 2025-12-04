import os
from typing import Dict

import torch
from torch import nn
from segment_anything import sam_model_registry
from PSPNet import PSPNetMaskDecoder
from sam_mona_image_encoder import Mona_SAM
class BaseModel(nn.Module):
    def __init__(
            self,
            n_channels,
            n_classes,
            bilinear,
            rank: int = 4 
         ):
        super(BaseModel, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        current_dir = os.path.dirname(os.path.abspath(__file__))

        checkpoint_path = os.path.join(current_dir, "checkpoint/sam_vit_b.pth")

        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)

        self.image_encoder = Mona_SAM(sam)

        self.mask_decoder = PSPNetMaskDecoder(
            num_classes=self.n_classes
        )

    def save_lora_parameters(self, lora_checkpoint_path: str):

        self.image_encoder.save_lora_parameters(lora_checkpoint_path)

    def load_lora_parameters(self, lora_checkpoint_path: str):

        self.image_encoder.load_lora_parameters(lora_checkpoint_path)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        image_embeddings = self.image_encoder(x)

        masks = self.mask_decoder(image_embeddings)

        return {
            "out": masks
        }
