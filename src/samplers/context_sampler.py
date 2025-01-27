import torch

from torch import Tensor
from typing import List, Tuple


class CtxBatchSampler:
    """Inspired by PrototypicalBatchSampler
     
    Yield a batch of indexes at each iteration. Indexes are calculated by keeping in account 'classes_per_it' 
    and 'num_samples'. At every iteration the batch indexes will refer to 'num_samples' for random 'classes_per_it'.
    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    """

    def __init__(self, labels: List[int], classes_per_it: int, n_samples_cls: int, iterations: int):
        """Initialize the CtxBatchSampler object
        
        Args:
            - labels (List[int]): ALL the labels for the current dataset. Samples indexes are inferred from here.
            - classes_per_it (int): number of random classes for each iteration
            - num_samples (int): number of samples for each iteration for each class
            - iterations (int): number of iterations (episodes) per epoch
        """
        
        super().__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = n_samples_cls
        self.iterations = iterations

        self.classes, self.counts = torch.unique(torch.tensor(self.labels, dtype=torch.long), return_counts=True)

        # indexes is a matrix of shape [classes X max(elements per class)]
        # each row contains the absolute indexes of the position for the current class sample found in self.labels
        self.indexes = torch.empty((len(self.classes), max(self.counts)), dtype=torch.int) * torch.nan
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = int(torch.argwhere(self.classes == label).item())
            self.indexes[label_idx, torch.where(torch.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        """yield a batch of indexes"""
        
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for _ in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                label_idx = int(torch.argwhere(self.classes == c).item())
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        return self.iterations
    
    @staticmethod
    def split_sq(
        recons: Tensor,
        target: Tensor,
        n_way: int,
        n_support: int,
        n_query: int,
        shuffle: bool=False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # map in [0, n_way-1] range
        classes = torch.unique(target)
        m_bool = (target == classes.unsqueeze(1))
        target = torch.max(m_bool, dim=0).indices.to(torch.int64)
        
        if not n_way == len(classes):
            raise ValueError(f"number of unique classes ({len(classes)}) must match config n_way ({n_way})")
        
        if not target.size(0) // len(classes) == n_support + n_query:
            raise ValueError(f"({target.size(0) // len(classes)}) != support ({n_support}) + query ({n_query})")
        
        class_idx = torch.stack(list(map(lambda x: torch.where(target == x)[0], torch.arange(n_way)))) # [n_way, s+q]
        support_idxs, query_idxs = torch.split(class_idx, [n_support, n_query], dim=1)

        support_set = recons[support_idxs.flatten()]
        query_set = recons[query_idxs.flatten()]
        support_labels = target[support_idxs.flatten()]
        query_labels = target[query_idxs.flatten()]

        if shuffle:
            perm_support = torch.randperm(support_set.size(0))
            support_set = support_set[perm_support]
            support_labels = support_labels[perm_support]
            perm_query = torch.randperm(query_set.size(0))
            query_set = query_set[perm_query]
            query_labels = query_labels[perm_query]

        return support_set, support_labels, query_set, query_labels