import os
import torch

from glob import glob
from typing import List, Set, Tuple, Optional

from .dataset_fsl import FewShotDataset
from ...utils.config_parser import DatasetConfig
from ...utils.tools import Logger, Tools
from ....config.consts import General as _CG

class Cub(FewShotDataset):
    """CUB 200 2011

    The orginal dataset train/test split does not account for a validation set, and most importantly it does not split
    train and test classes: we want to classify unseen classes, not unseen instances! Meta-dataset provides these splits
    instead.

    Not all classes have the same number of samples: 144/200 classes have exactly 60 samples. The remaining classes
    have less samples.

    SeeAlso:
        [FSL dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/)
        [split](https://github.com/google-research/meta-dataset/blob/main/meta_dataset/dataset_conversion/splits/cu_birds_splits.json)
    """

    N_CLASSES = 200
    N_CLASS_TRAIN = 140
    N_CLASS_VAL = 30
    N_CLASS_TEST = 30
    N_IMAGES = 11788
    IMG_DIR = "images"

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
        self.img_dir_path = Tools.validate_path(os.path.join(self.dataset_config.dataset_path, self.IMG_DIR))
        self.split_dir = Tools.validate_path(os.path.join(os.path.abspath(__file__).rsplit("src", 1)[0], "splits", "cub"))
        
        super().__init__(dataset_config)

    def get_image_list(self, filt: Optional[List[str]]) -> List[str]:
        img_list = glob(os.path.join(self.img_dir_path, "*", "*.jpg"))
        return img_list
    
    def split_method(self) -> Tuple[Set[str], Set[str], Set[str]]:
        obj = Tools.read_json(os.path.join(self.split_dir, "cu_birds_splits.json"))
        
        class_train = set(obj.get("train"))
        class_val = set(obj.get("valid"))
        class_test = set(obj.get("test"))
        
        return class_train, class_val, class_test

    def expected_length(self) -> int:
        return self.N_IMAGES
