from typing import Optional

from src.models.model import Model
from src.models.context.camlu import Camlu
from src.models.extractors.custom import Resnet50FE
from src.models.extractors.plbolts import PlboltsFe
from src.models.extractors.timm_ext import TimmFeatureExtractor
from src.utils.config_parser import Config


class ModelBuilder:

    @staticmethod
    def load_model(config: Config, out_classes: Optional[int]=None) -> Model:
        model_name = config.model.model_name.lower().replace(" ", "")
        if "camlu" in model_name:
            _, fe_lib, fe_name = model_name.rsplit(":", -1)
            
            # timm feature extractors
            if fe_lib == "timm":
                
                # ViT-CLIP
                if fe_name == "vit_base_patch16_clip_224":
                    fe_name = f"{fe_name}.openai" if config.model.pretrained else fe_name
                
                # resnet50
                if fe_name == "resnet50":
                    fe_name = f"{fe_name}.a1_in1k" if config.model.pretrained else fe_name

                # else: directly use the name provided in the config file

                extractor = TimmFeatureExtractor(
                    config,
                    fe_name,
                    pretrained=config.model.pretrained,
                    in_chans=3,
                    pooled=True,
                    mean=config.dataset.dataset_mean,
                    std=config.dataset.dataset_std
                )
                return Camlu(config, extractor)
            
            # pl_bolts feature extractor, e.g. camlu:plbolts:swav
            if fe_lib in ("plbolts", "pl_bolts"):
                extractor = PlboltsFe(config, fe_name)
                return Camlu(config, extractor)

            # custom pre-trained feature extractors to be put here
            if fe_lib == "custom":
                if fe_name == "resnet50":
                    n_classes = 1000 if config.dataset.dataset_type == "episodic_imagenet1k" else 964
                    extractor = Resnet50FE(config, n_classes)
                    return Camlu(config, extractor)
        
        raise ValueError(f"The name of the model does not match all the criteria.")