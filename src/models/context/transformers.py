from torch import nn, Tensor
from typing import Callable
from functools import partial
from collections import OrderedDict
from torchvision.models.vision_transformer import MLPBlock

from src.utils.config_parser import Model as ConfigModel


class VitEncoder(nn.Module):

    def __init__(self, model_config: ConfigModel, norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.context_config = model_config.context
        self.dropout = nn.Dropout(model_config.dropout)
        
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(self.context_config.n_layers):
            layers[f"encoder_layer_{i}"] = VitEncoderBlock(model_config, norm_layer)
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(self.context_config.hidden_dim)

    def forward(self, x: Tensor):
        if not x.dim() == 3:
            raise ValueError(f"Expected (batch_size, seq_length, hidden_dim), got {x.dim} as {x.shape}")
        
        out = self.dropout(x)
        out = self.layers(out)
        out = self.ln(out)
        return out
    

class VitEncoderBlock(nn.Module):
    """One ViT encoder block
    
    SeeAlso:
        [image src](https://www.researchgate.net/publication/360018583/figure/fig4/AS:1147522454159364@1650602081646/Overall-architecture-of-the-ViT-encoder.ppm)
    """

    def __init__(self, config_model: ConfigModel, norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.config_context = config_model.context
        
        # Attention block
        self.ln_1 = norm_layer(self.config_context.hidden_dim)
        self.dropout = nn.Dropout(config_model.dropout)
        self.self_attention = nn.MultiheadAttention(
            self.config_context.hidden_dim,
            self.config_context.n_heads,
            dropout=self.config_context.attention_dropout,
            batch_first=True
        )

        # MLP block
        self.ln_2 = norm_layer(self.config_context.hidden_dim)
        self.mlp = MLPBlock(self.config_context.hidden_dim, self.config_context.mlp_dim, config_model.dropout)

    def forward(self, x: Tensor):
        if not x.dim() == 3:
            raise ValueError(f"Expected (batch_size, seq_length, hidden_dim), got {x.dim} as {x.shape}")
        
        residual_1 = x
        out_1 = self.ln_1(x)
        out_1, _ = self.self_attention(query=out_1, key=out_1, value=out_1, need_weights=False)
        out_1 = self.dropout(out_1)
        out_1 = out_1 + residual_1

        residual_2 = out_1
        out_2 = self.ln_2(out_1)
        out_2 = self.mlp(out_2)
        return out_2 + residual_2