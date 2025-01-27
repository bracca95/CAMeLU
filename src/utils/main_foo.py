import os
import wandb
import torch
import numpy as np

from copy import deepcopy
from typing import Tuple, List

from src.utils.config_parser import Config, read_from_json, write_to_json
from lib.libdataset.src.utils.tools import Logger, Tools
from lib.libdataset.src.datasets.dataset import DatasetLauncher, DatasetWrapper
from lib.libdataset.src.datasets.dataset_utils import DatasetBuilder
from lib.libdataset.config.consts import CustomDatasetConsts as _CDC


class MainFoo:
    
    @staticmethod
    def init_wandb(config: Config):
        ## start program
        split_name = config.experiment_name.split(":")

        if len(split_name) == 1:
            if split_name[0] == "disabled":
                wandb_mode = split_name[0]
                exp_name = split_name[0]
            else:
                wandb_mode = "online"
                exp_name = split_name[0]
                
        if len(split_name) > 1:
            wandb_mode = split_name[1]
            exp_name = split_name[0]
        
        wandb.init(
            mode=wandb_mode,
            project=exp_name,
            config={
                "learning_rate": config.train_test.learning_rate,
                "architecture": config.model.model_name,
                "dataset": config.dataset.dataset_type,
                "epochs": config.train_test.epochs,
            }
        )

    @staticmethod
    def set_seed(config: Config):
        # Set seed for CPU
        SEED = config.seed
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # Set seed for GPU if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            # CuDNN algorithms may be non-deterministic. This can be fixed by setting:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def ds_mng(config: Config, config_path: str, dataset_wrapper: DatasetWrapper) -> Tuple[Config, List[DatasetWrapper]]:
        # compute mean and variance of the dataset if not done yet
        if config.dataset.normalize and config.dataset.dataset_mean is None and config.dataset.dataset_std is None:
            Logger.instance().warning("No mean and std set: computing and storing values.")
            mean, std = DatasetLauncher.compute_mean_std(dataset_wrapper.train_dataset, config.dataset.dataset_type)
            config.dataset.dataset_mean = mean.tolist()
            config.dataset.dataset_std = std.tolist()
            write_to_json(config, os.getcwd(), config_path)
            
            # reload and override
            config = read_from_json(config_path)
            dataset_wrapper = DatasetBuilder.load_dataset(config.dataset)
        
        # multi dataset
        dw_list = []
        if type(config.dataset.dataset_id) is list and len(config.dataset.dataset_id) > 0:
            
            # coco (11111) and fungi (22222)
            for did in config.dataset.dataset_id:
                if did == _CDC.EpisodicCoco:
                    curr_ds_type = "episodic_coco"
                    curr_rel_pth = f"mscoco{os.sep}episodic_coco"
                if did == _CDC.Fungi:
                    curr_ds_type = "fungi"
                    curr_rel_pth = "fungi"

                Logger.instance().debug(f"Trying to also use {curr_ds_type} ({did}) as training dataset..")
                dirname = os.path.dirname(config.dataset.dataset_path)
                try:
                    ds_path = Tools.validate_path(os.path.join(dirname, curr_rel_pth))
                except FileNotFoundError as fnf:
                    Logger.instance().critical(
                        f"{fnf}. {curr_ds_type} supposed to be in {os.path.join(dirname, curr_rel_pth)}. " +
                        f"More info in lib/libdataset/src/datasets/fsl/{curr_ds_type}.py"
                    )
                ds_config = deepcopy(config)
                ds_config.dataset.dataset_path = ds_path
                ds_config.dataset.dataset_type = curr_ds_type
                ds_config.dataset.dataset_splits = [1.0, 0.0, 0.0] # use only for training
                curr_dataset = DatasetBuilder.load_dataset(ds_config.dataset)
                dw_list.append(curr_dataset)
        
        dw_list.insert(0, dataset_wrapper)
        Logger.instance().debug(f"Training over {len(dw_list)} datasets")

        return config, dw_list