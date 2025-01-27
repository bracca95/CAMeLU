from abc import abstractmethod
from typing import Optional, List, Tuple, Iterator
from torch.utils.data import DataLoader

from src.models.model import Model
from src.routines.caml_routine import CamlRoutine, Trainloaders
from src.samplers.context_sampler import CtxBatchSampler
from src.utils.config_parser import TrainTest as TrainTestConfig
from lib.libdataset.src.datasets.dataset import DatasetWrapper
from lib.libdataset.config.consts import General as _CG


class CamluStrongRoutine(CamlRoutine):

    def __init__(self, train_test_config: TrainTestConfig, model: Model, dataset_wrapper: DatasetWrapper):
        super().__init__(train_test_config, model, dataset_wrapper)

    def init_trainloaders(self) -> Trainloaders:
        return Trainloaders(self.init_loader(self.train_str))
    
    def get_iters(self, trainloaders: Trainloaders, epoch: int) -> Tuple[List[Iterator], Optional[List[Iterator]]]:
        return [iter(trainloaders.trainloader)], None
    
    def choose_task_dataset(
        self,
        iter_list_1: List[Iterator],
        iter_list_2: Optional[List[Iterator]],
        dataset_lengths: List[int]
    ) -> Tuple[Iterator, Optional[Iterator]]:
        return iter_list_1[0], None

    def get_img_lbl_from_loaders(self, loader_1: Iterator, loader_2: Optional[Iterator]):
        x, y = next(loader_1)
        
        # get relevant value
        n_way = self._model_config.context.n_way
        k_shot = self._model_config.context.k_shot
        k_query = self._model_config.context.k_query
        img_size = self.model.config.dataset.image_size
        
        # reshape
        b = n_way * (k_shot + k_query)
        x = x.reshape(b, 3, img_size, img_size)
        y = y.unsqueeze(1).repeat(1, k_shot + k_query).flatten()
        
        return x, y
        
    def init_loader(self, split_set: str, augment: Optional[str]=None) -> Optional[DataLoader]:
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
    
        # whole training must contain strong augmentations
        sampler = CtxBatchSampler(
            labels=current_dataset.label_list,
            classes_per_it=self._model_config.context.n_way,
            n_samples_cls=1,
            iterations=self.episodes
        )

        return DataLoader(current_dataset, batch_sampler=sampler, num_workers=self.train_test_config.num_workers)
