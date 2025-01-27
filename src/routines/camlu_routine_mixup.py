import torch

from copy import deepcopy
from torch import Tensor
from typing import Optional, Tuple, List, Iterator
from torch.utils.data import DataLoader

from src.models.model import Model
from src.routines.caml_routine import CamlRoutine, Trainloaders
from src.samplers.context_sampler import CtxBatchSampler
from src.utils.config_parser import TrainTest as TrainTestConfig
from lib.libdataset.src.datasets.dataset import DatasetWrapper
from lib.libdataset.config.consts import General as _CG


class CamluMixupRoutine(CamlRoutine):

    def __init__(self, train_test_config: TrainTestConfig, model: Model, dataset_wrapper: DatasetWrapper):
        super().__init__(train_test_config, model, dataset_wrapper)

    def init_trainloaders(self) -> Trainloaders:
        return Trainloaders(
            None,
            [self.init_loader(self.train_str, augment="support")],
            [self.init_loader(self.train_str, augment="query")]
        )
    
    def get_iters(self, trainloaders: Trainloaders, epoch: int) -> Tuple[List[Iterator], Optional[List[Iterator]]]:
        return [iter(trainloaders.supports[0])], [iter(trainloaders.queries[0])]
    
    def choose_task_dataset(
        self,
        iter_list_1: List[Iterator],
        iter_list_2: Optional[List[Iterator]],
        dataset_lengths: List[int]
    ) -> Tuple[Iterator, Optional[Iterator]]:
        return iter_list_1[0], iter_list_2[0]

    def get_img_lbl_from_loaders(self, loader_1: Iterator, loader_2: Optional[Iterator]):
        """Split support/query batch for SSL

        Custom method to split the support batch (n_way, k_shot, n_chans, img_size, img_size) from the query batch
        (n_way * k_shot, n_chans, img_size, img_size). Then reshaped together to be consistent with the rest of the
        pipeline. During SSL, the support and query sets come as a zip, so each of them is a list of Tensor(s) with
        [img, label]. 

        Args:
            loader_1 (Iterator): support interator
            loader_2 (Iterator): query iterator. CANNOT be null in this case!

        Returns:
            img, label (Tuple[Tensor, Tensor])
        """

        if loader_2 is None:
            raise ValueError(f"the second loader cannot be null in this case!")

        xs_full, ys_full = next(loader_1)
        xq, _ = next(loader_2)

        # get relevant value
        n_way = self._model_config.context.n_way
        k_shot = self._model_config.context.k_shot
        k_query = self._model_config.context.k_query
        img_size = self.model.config.dataset.image_size
        
        # support full (contains also queries)
        bs_s = n_way * k_shot
        bs_q = n_way * k_query
        ys_full = ys_full.unsqueeze(1).repeat(1, k_shot+k_query)

        # reserve n_way * k_shot supports for queries mixup
        xs, xsq = torch.split(xs_full, [k_shot, k_query], dim=1)
        ys, ysq = torch.split(ys_full, [k_shot, k_query], dim=1)

        # reshape
        xs = xs.reshape(bs_s, 3, img_size, img_size)
        xsq = xsq.reshape(bs_q, 3, img_size, img_size)
        ys = ys.flatten()
        ysq = ysq.flatten()

        # query and their augmentation
        xq, yq = self.augment_query(xq, xsq, ysq)
        del xsq, ysq

        # put together
        x = torch.cat((xs, xq), dim=0)
        y = torch.cat((ys, yq), dim=0)
        
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
        
        if augment == "support":
            support = self.deepcopy_dataset(augment)
            support_train = getattr(support, f"{train_set}_dataset")
            sampler = CtxBatchSampler(
                labels=current_dataset.label_list,
                classes_per_it=self._model_config.context.n_way,
                n_samples_cls=1,
                iterations=self.episodes
            )
            return DataLoader(support_train, batch_sampler=sampler, num_workers=self.train_test_config.num_workers)
        
        if augment == "query":
            query = self.deepcopy_dataset(augment)
            query_train = getattr(query, f"{train_set}_dataset")
            sampler = CtxBatchSampler(
                labels=current_dataset.label_list,
                classes_per_it=self._model_config.context.n_way * self._model_config.context.k_query,
                n_samples_cls=1,
                iterations=self.episodes
            )
            return DataLoader(query_train, batch_sampler=sampler, num_workers=self.train_test_config.num_workers)
            
        raise ValueError(f"No condition met for `augment_online`. Check it again or use `null`")
    
    def deepcopy_dataset(self, augment: str):
        from lib.libdataset.src.datasets.dataset_utils import DatasetBuilder

        config_copy = deepcopy(self.model.config)
        config_copy.dataset.augment_online = [augment]
        config_copy.serialize()
        dataset_copy = DatasetBuilder.load_dataset(config_copy.dataset)
            
        return dataset_copy

    @staticmethod
    def augment_query(xq: Tensor, xs: Tensor, ys: Tensor) -> Tuple[Tensor, Tensor]:
        """Augment queries via mixup
        
        Args:
            xq (Tensor): query images
            xs (Tensor): support images
            ys (Tensor): support labels

        Returns:
            modified query images, labels (Tuple[Tensor, Tensor])
        """
        
        if not xq.size(0) % xs.size(0) == 0:
            raise ValueError(f"query Tensor shape {(xq.shape)} must be n times support shape ({xs.size(0)})")

        # use support images to build mixed queries
        lam = torch.distributions.Uniform(0.01, 0.499).sample((1,)).item()
        shuffle = torch.randperm(xs.size(0))

        # mixup: the support should contribute more to be considered the correct label
        xq = lam * xq + (1 - lam) * xs[shuffle]
        yq = ys[shuffle]

        return xq, yq