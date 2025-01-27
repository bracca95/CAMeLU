import torch

from copy import deepcopy
from typing import Optional, Tuple, List, Iterator
from torch.utils.data import DataLoader

from src.models.model import Model
from src.utils.config_parser import Config
from src.routines.camlu_routine_mixup import CamluMixupRoutine
from src.routines.caml_routine import Trainloaders
from src.samplers.context_sampler import CtxBatchSampler
from src.utils.config_parser import TrainTest as TrainTestConfig
from lib.libdataset.src.datasets.dataset import DatasetWrapper
from lib.libdataset.src.datasets.fsl.dataset_fsl import FewShotDataset
from lib.libdataset.src.utils.tools import Logger, Tools
from lib.libdataset.config.consts import General as _CG


class CamluMixupMultiRoutine(CamluMixupRoutine):

    def __init__(self, train_test_config: TrainTestConfig, model: Model, dataset_wrapper: DatasetWrapper, *add_ds):
        super().__init__(train_test_config, model, dataset_wrapper)
        self.dataset_list = [self.dataset_wrapper] + list(add_ds)

    def init_trainloaders(self) -> Trainloaders:
        support_loaders = [
            self.init_loader(self.train_str, "support", i) for i in range(len(self.dataset_list))
        ]

        query_loaders = [
           self.init_loader(self.train_str, "query", i) for i in range(len(self.dataset_list))
        ]
        
        return Trainloaders(None, support_loaders, query_loaders)
    
    def get_iters(self, trainloaders: Trainloaders, epoch: int) -> Tuple[List[Iterator], Optional[List[Iterator]]]:
        iter_list_1 = [iter(trainloaders.supports[i]) for i in range(len(trainloaders.supports))]
        iter_list_2 = [iter(trainloaders.queries[i]) for i in range(len(trainloaders.queries))]
        
        return iter_list_1, iter_list_2

    def choose_task_dataset(
        self,
        iter_list_1: List[Iterator],
        iter_list_2: Optional[List[Iterator]],
        dataset_lengths: List[int]
    ) -> Tuple[Iterator, Optional[Iterator]]:
        
        sample = torch.randint(0, dataset_lengths[-1], (1,))
        for i, ds_max in enumerate(dataset_lengths):
            if sample < ds_max:
                return iter_list_1[i], iter_list_2[i]
    
    def init_loader(self, split_set: str, augment: Optional[str]=None, *args) -> Optional[DataLoader]:
        train_set, val_set, test_set = _CG.DEFAULT_SUBSETS
        current_dataset = getattr(self.dataset_wrapper, f"{split_set}_dataset")
        
        if current_dataset is None:
            return None

        # for val/test split return the default behaviour
        if not split_set == train_set:
            sampler = CtxBatchSampler(
                labels=current_dataset.label_list,
                classes_per_it=self._model_config.context.n_way,
                n_samples_cls=self._model_config.context.k_shot + self._model_config.context.k_query,
                iterations=self.episodes
            )
        
            return DataLoader(current_dataset, batch_sampler=sampler, num_workers=self.train_test_config.num_workers)
        
        # from here on: train support/query only
        pos = list(args)[0]
        if augment == "support":
            support = self.deepcopy_dataset(self.model.config, self.dataset_list[pos], augment)
            support_train = getattr(support, f"{train_set}_dataset")
            sampler = CtxBatchSampler(
                labels=support_train.label_list,
                classes_per_it=self._model_config.context.n_way,
                n_samples_cls=1,
                iterations=self.episodes
            )
            return DataLoader(support_train, batch_sampler=sampler, num_workers=self.train_test_config.num_workers)
        
        if augment == "query":
            query = self.deepcopy_dataset(self.model.config, self.dataset_list[pos], augment)
            query_train = getattr(query, f"{train_set}_dataset")
            sampler = CtxBatchSampler(
                labels=query_train.label_list,
                classes_per_it=self._model_config.context.n_way * self._model_config.context.k_query,
                n_samples_cls=1,
                iterations=self.episodes
            )
            return DataLoader(query_train, batch_sampler=sampler, num_workers=self.train_test_config.num_workers)
            
        raise ValueError(f"No condition met for `augment_online`. Check it again or use `null`")
    
    @staticmethod
    def deepcopy_dataset(config: Config, dataset_wrapper: FewShotDataset, augment: Optional[str]):
        from lib.libdataset.src.datasets.dataset_utils import DatasetBuilder
        
        augment_param = augment if augment is None else [augment]

        # copy config file and updated config for the specific dataset (main or additional)
        whole_config_copy = deepcopy(config)
        curr_ds_config_copy = deepcopy(dataset_wrapper.dataset_config)

        # update current dataset configs (augment online and meta-album mds)
        curr_ds_config_copy.augment_online = augment_param
        curr_ds_config_copy.dataset_id = [getattr(dataset_wrapper, "did")] if hasattr(dataset_wrapper, "did") else None   
        
        # finally replace the whole config
        whole_config_copy.dataset = curr_ds_config_copy

        # serialize new config
        whole_config_copy.serialize()
        dataset_copy = DatasetBuilder.load_dataset(whole_config_copy.dataset)
            
        return dataset_copy
