from typing import Union
from torch.utils.data import Dataset

from .dataset import DatasetWrapper
from .fsl.episodic_imagenet import EpisodicImagenet, EpisodicImagenetValCifar, EpisodicImagenetValCub, EpisodicImagenetValAircraft
from .fsl.episodic_imagenet1k import EpisodicImagenet1k
from .fsl.episodic_coco import EpisodicCoco
from .fsl.miniimagenet import MiniImagenet
from .fsl.cifar import CifarFs
from .fsl.cub import Cub
from .fsl.fungi import Fungi
from .fsl.aircraft import Aircraft
from .fsl.meta_inat import Metainat
from ..utils.config_parser import DatasetConfig
from ..utils.tools import Logger


class DatasetBuilder:

    @staticmethod
    def load_dataset(dataset_config: DatasetConfig) -> Union[DatasetWrapper, Dataset]:
        if dataset_config.dataset_type == "episodic_imagenet":
            Logger.instance().info("Loading dataset EpisodicImagenet (type FewShotDataset)")
            return EpisodicImagenet(dataset_config)
        elif dataset_config.dataset_type == "episodic_imagenet_val_cifar":
            Logger.instance().info("Loading dataset EpisodicImagenetValCifar (type FewShotDataset)")
            return EpisodicImagenetValCifar(dataset_config)
        elif dataset_config.dataset_type == "episodic_imagenet_val_cub":
            Logger.instance().info("Loading dataset EpisodicImagenetValCub (type FewShotDataset)")
            return EpisodicImagenetValCub(dataset_config)
        elif dataset_config.dataset_type == "episodic_imagenet_val_aircraft":
            Logger.instance().info("Loading dataset EpisodicImagenetValAircraft (type FewShotDataset)")
            return EpisodicImagenetValAircraft(dataset_config)
        elif dataset_config.dataset_type == "episodic_imagenet1k":
            Logger.instance().info("Loading dataset EpisodicImagenet1k (type FewShotDataset)")
            return EpisodicImagenet1k(dataset_config)
        elif dataset_config.dataset_type == "episodic_coco":
            Logger.instance().info("Loading dataset EpisodicCoco (type FewShotDataset)")
            return EpisodicCoco(dataset_config)
        elif dataset_config.dataset_type == "miniimagenet":
            Logger.instance().info("Loading dataset Mini Imagenet (type FewShotDataset)")
            return MiniImagenet(dataset_config)
        elif dataset_config.dataset_type == "cifar_fs":
            Logger.instance().info("Loading dataset CIFAR-FS (type FewShotDataset)")
            return CifarFs(dataset_config)
        elif dataset_config.dataset_type == "cub":
            Logger.instance().info("Loading dataset CUB (type FewShotDataset)")
            return Cub(dataset_config)
        elif dataset_config.dataset_type == "fungi":
            Logger.instance().info("Loading dataset Fungi (type FewShotDataset)")
            return Fungi(dataset_config)
        elif dataset_config.dataset_type == "aircraft":
            Logger.instance().info("Loading dataset Aircraft (type FewShotDataset)")
            return Aircraft(dataset_config)
        elif dataset_config.dataset_type == "meta_inat":
            Logger.instance().info("Loading dataset Meta-iNat (type FewShotDataset)")
            return Metainat(dataset_config)
        else:
            raise ValueError(
                "values allowed: {`episodic_imagenet`, `episodic_imagenet1k`, `episodic_coco`, " +
                "`miniimagenet`, `cub`, `fungi`, `aircraft`, `meta_inat`, for dataset_type.\n" +
                "`episodic_imagenet` can also be run with other evaluation datasets: append " +
                "(_val_cifar, _val_cub, _val_aircraft)"
            )
