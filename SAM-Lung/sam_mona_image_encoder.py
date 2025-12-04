import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from copy import deepcopy
import torch.nn.functional as F


INNER_DIM = 64

class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x
        x = self.projector(x)

        return identity + x

class Mona(nn.Module):
    def __init__(self, in_dim, factor=4):
        super().__init__()
        self.project1 = nn.Linear(in_dim, INNER_DIM)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(INNER_DIM, in_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.adapter_conv = MonaOp(INNER_DIM)
        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):

        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax
        project1 = self.project1(x)


        b, n, c = project1.shape
        h, w = hw_shapes
        assert h * w == n, f"hw_shapes={hw_shapes} 与序列长度n={n}不匹配"
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2

class WrappedSwinBlock(nn.Module):
    def __init__(self, original_block, embed_dim):
        super().__init__()
        self.original_block = original_block
        self.mona1 = Mona(embed_dim)
        self.mona2 = Mona(embed_dim)

    def forward(self, x, hw_shape):

        identity = x

        x = self.original_block.norm1(x)

        B, N, C = x.shape
        H, W = hw_shape
        x = x.view(B, H, W, C)  
        x = self.original_block.attn(x)        

        x = x.view(B, -1, C)

        x = self.mona1(x, hw_shapes=hw_shape)  
        x = identity + x                      

        identity = x
        x = self.original_block.norm2(x)
        x = self.original_block.mlp(x)          
        x = self.mona2(x, hw_shapes=hw_shape)  
        x = identity + x                       

        return x

class Mona_SAM(nn.Module):
    def __init__(self, sam_model, mona_layers=None):
        super().__init__()
        self.sam = sam_model
        for param in self.sam.parameters():
            param.requires_grad = False

        embed_dim = self.sam.image_encoder.blocks[0].attn.qkv.in_features
        num_blocks = len(self.sam.image_encoder.blocks)
        self.mona_layers = list(range(num_blocks)) if mona_layers is None else mona_layers

        for i in self.mona_layers:
            original_block = self.sam.image_encoder.blocks[i]
            self.sam.image_encoder.blocks[i] = WrappedSwinBlock(deepcopy(original_block), embed_dim)

    def get_mona_parameters(self):
            params = {}
            for name, module in self.named_modules():
                if isinstance(module, Mona):
                    for param_name, param in module.named_parameters():
                        full_key = f"{name}.{param_name}"
                        params[full_key] = param
                elif isinstance(module, WrappedSwinBlock):
                    for mona_name in ["mona1", "mona2"]:
                        mona_module = getattr(module, mona_name, None)
                        if isinstance(mona_module, Mona):
                            for param_name, param in mona_module.named_parameters():
                                full_key = f"{name}.{mona_name}.{param_name}"
                                params[full_key] = param
            return params

    def save_mona_parameters(self, filename: str):
            mona_params = self.get_mona_parameters()
            torch.save(mona_params, filename)

    def load_mona_parameters(self, filename: str, strict=True):
            device = next(self.parameters()).device 
            saved_params = torch.load(filename, map_location=device)
            current_mona_params = self.get_mona_parameters()

            for full_name, param in current_mona_params.items():
                if full_name in saved_params:
                    param.data.copy_(saved_params[full_name])
                elif strict:
                    raise KeyError(f"未找到参数: {full_name}")

    def forward(self, x):
        x = self.sam.image_encoder.patch_embed(x)

        x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        current_h, current_w = H, W

        for blk in self.sam.image_encoder.blocks:
            if isinstance(blk, WrappedSwinBlock):

                x = blk(x, hw_shape=(current_h, current_w))

            elif 'PatchMerging' in str(type(blk)):  
                x = blk(x, (current_h, current_w))
                current_h, current_w = current_h // 2, current_w // 2  
            else:
                x = blk(x)

        x = x.transpose(1, 2).reshape(B, -1, current_h, current_w)
        x = self.sam.image_encoder.neck(x)
        return x