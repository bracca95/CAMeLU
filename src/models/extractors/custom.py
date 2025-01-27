import sys
import torch

from torchvision import models
from typing import Optional, List
from src.models.model import Model
from src.utils.config_parser import Config
from lib.libdataset.src.utils.tools import Logger, Tools


class Resnet50FE(Model):

    def __init__(self, config: Config, out_classes: Optional[int]=1000):
        super().__init__(config)
        self.out_classes = out_classes

        # load bare model for training
        self.m = models.resnet50(weights=None)
        self.m.fc = torch.nn.Linear(self.m.fc.in_features, self.out_classes)
        
        # if you want to use as a feature extractor, after training
        if self.config.model.freeze and self.config.model.pretrained:
            weights_path = self.check_weights_path("resources/resnet50.pth")
            state_dict = torch.load(weights_path)
            fix_state_dict = { key.replace("m.", "", 1): state_dict[key] for key in state_dict.keys() }
            self.m.load_state_dict(fix_state_dict)
            self.m.fc = torch.nn.Identity()

    def forward(self, x):
        if self.config.model.freeze and self.config.model.pretrained:
            return self.__fwd_eval(x)
        
        return self.m(x)
    
    def __fwd_eval(self, x):
        self.m = self.m.eval()
        for p in self.m.parameters(): p.requires_grad = False
        return self.m(x)
