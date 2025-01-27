import math
import torch

from torch import nn, Tensor
from torch.nn import functional as F

from src.models.model import Model
from src.models.context.transformers import VitEncoder
from src.samplers.context_sampler import CtxBatchSampler as Ctx
from src.utils.config_parser import Config
from lib.libdataset.config.consts import General as _CG


class Camlu(Model):

    def __init__(self, config: Config, extractor: Model):
        super().__init__(config)
        self.config = config
        self.ctx_cfg = self.config.model.context
        self.n_way = self.ctx_cfg.n_way
        self.k_shot = self.ctx_cfg.k_shot
        self.k_query = self.ctx_cfg.k_query

        # feature extractor: in this case the parameter were freezed during instantiation
        self.extractor = extractor.to(_CG.DEVICE)
        self.clip_hdim = self.extractor.get_out_size(1)
        
        # transformer encoder
        self.encoder = VitEncoder(self.config.model)
        
        # label encoder
        class_hdim = self.check_hdim(self.ctx_cfg.hidden_dim, self.clip_hdim)
        self.elmes_scale = nn.Parameter(torch.ones(1))
        self.label_elmes = nn.Parameter(torch.empty(self.n_way, class_hdim, device=_CG.DEVICE))
        torch.nn.init.kaiming_uniform_(self.label_elmes, a=math.sqrt(5))
        self.unk_emb = nn.Parameter(torch.zeros(1, 1, class_hdim))      # unknown embedding for the label

        # output
        self.output_proj = nn.Linear(in_features=self.ctx_cfg.hidden_dim, out_features=self.n_way, bias=False)

    def forward(self, x: Tensor, y: Tensor):
        feat = self.extractor(x)
        xs_emb, ys, xq_emb, yq = Ctx.split_sq(feat, y, self.n_way, self.k_shot, self.k_query, shuffle=True)
        
        # feature sequence (repeat the support |B| times for each query example)
        support = xs_emb.reshape(1, self.n_way * self.k_shot, self.clip_hdim).repeat(xq_emb.size(0), 1, 1)
        query = xq_emb.reshape(-1, 1, self.clip_hdim)
        feature_seq = torch.cat([query, support], dim=1)

        # label sequence
        ys_one_hot = F.one_hot(ys.unsqueeze(0), num_classes=self.n_way).to(torch.float32)
        ys_emb = ys_one_hot @ (self.elmes_scale * self.label_elmes)
        batched_ys_emb = torch.cat([self.unk_emb, ys_emb], dim=1).repeat(query.size(0), 1, 1)
        
        # cat features and labels to create demonstrations
        demonstrations = torch.cat([feature_seq, batched_ys_emb], dim=-1)
        seq = self.encoder.forward(demonstrations)

        query = seq[:, 0, :]
        logits = self.output_proj(query)
        return logits, yq
    
    @staticmethod
    def check_hdim(cfg_hidden_dim: int, fe_hdim: int) -> int:
        """Check hidden dimensions

        Check that the hidden dimension in config is greater than feature extractor's hidden dim. This is because the
        hidden dimension defined in the config must take into account [feature extractor + the positional] encoding dim,
        which are concatenated. The positional encoding is not fixed, rather computed as the difference between the
        config and the FE, hence it must be greater than 0.

        Args:
            cfg_hidden_dim (int): what is set as `config.model.hidden_dim`
            fe_dim (int): feature extractor's hidden dim

        Returns:
            positional embedding's hidden dim (int)
        
        Raises:
            ValueError if cfg_hidden_dim - fe_hdim <= 0
        """
        
        class_hdim = cfg_hidden_dim - fe_hdim
        if class_hdim <= 0:
            msg = f"Set a hidden dimension ({cfg_hidden_dim}) that exceeds the clip " + \
                  f"(feature extractor) hdim ({fe_hdim})"
            raise ValueError(msg)
        
        return class_hdim