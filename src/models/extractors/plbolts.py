from pl_bolts.models.self_supervised import SwAV, SimCLR

from src.models.model import Model
from src.utils.config_parser import Config
from lib.libdataset.src.utils.tools import Logger


class PlboltsFe(Model):
    """pytorch lightning pre-trained models

    SeeAlso:
        - [pl_bolts doc](https://lightning-bolts.readthedocs.io/en/0.3.4/models_howto.html)
    """

    def __init__(self, config: Config, name: str):
        super().__init__(config)
        self.name = name

        if self.name.lower() == "swav":
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
            model = SwAV.load_from_checkpoint(weight_path, strict=True)
            self.m = model
        elif self.name.lower() == "simclr":
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
            model = SimCLR.load_from_checkpoint(weight_path, strict=False)
            self.m = model.encoder
        else:
            raise ValueError(f"Only 'swav' and 'simclr' are known.")

    def forward(self, x):
        if self.config.model.freeze:
            return self.__fwd_eval(x)
        
        if self.name.lower() == "simclr":
            return self.m(x)[0]

        return self.m(x)
    
    def __fwd_eval(self, x):
        self.m = self.m.eval()
        for p in self.m.parameters(): p.requires_grad = False
        
        if self.name.lower() == "simclr":
            return self.m(x)[0]

        return self.m(x)
