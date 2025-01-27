import torch

from typing import Optional, Tuple, List, Iterator
from torch.utils.data import DataLoader

from src.models.model import Model
from src.routines.caml_routine import CamlRoutine, Trainloaders
from src.routines.camlu_routine_mixup_multi import CamluMixupMultiRoutine as RUtils
from src.samplers.context_sampler import CtxBatchSampler
from src.utils.config_parser import TrainTest as TrainTestConfig
from lib.libdataset.src.datasets.dataset import DatasetWrapper
from lib.libdataset.src.utils.tools import Logger, Tools
from lib.libdataset.config.consts import General as _CG


class CamlRoutineMulti(CamlRoutine):

    def __init__(self, train_test_config: TrainTestConfig, model: Model, dataset_wrapper: DatasetWrapper, *add_ds):
        super().__init__(train_test_config, model, dataset_wrapper)
        self.dataset_list = [self.dataset_wrapper] + list(add_ds)

    def init_trainloaders(self) -> Trainloaders:
        multi_loader = [
            self.init_loader(self.train_str, None, i) for i in range(len(self.dataset_list))
        ]
        
        return Trainloaders(trainloader=multi_loader)
    
    def get_iters(self, trainloaders: Trainloaders, epoch: int) -> Tuple[List[Iterator], Optional[List[Iterator]]]:
        iter_list_1 = [iter(trainloaders.trainloader[i]) for i in range(len(trainloaders.trainloader))]
        return iter_list_1, None
    
    def choose_task_dataset(
        self,
        iter_list_1: List[Iterator],
        iter_list_2: Optional[List[Iterator]],
        dataset_lengths: List[int]
    ) -> Tuple[Iterator, Optional[Iterator]]:
        
        sample = torch.randint(0, dataset_lengths[-1], (1,))
        for i, ds_max in enumerate(dataset_lengths):
            if sample < ds_max:
                return iter_list_1[i], None
    
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

        pos = list(args)[0]
        select_ds = RUtils.deepcopy_dataset(self.model.config, self.dataset_list[pos], augment)
        current_dataset = getattr(select_ds, f"{train_set}_dataset")
        sampler = CtxBatchSampler(
            labels=current_dataset.label_list,
            classes_per_it=self._model_config.context.n_way,
            n_samples_cls=self._model_config.context.k_shot + self._model_config.context.k_query,
            iterations=self.episodes
        )
        return DataLoader(current_dataset, batch_sampler=sampler, num_workers=self.train_test_config.num_workers)