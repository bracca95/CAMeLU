import math
import torch

from abc import ABC, abstractmethod
from typing import Optional, Deque
from collections import deque
from torch.optim.lr_scheduler import LambdaLR

from src.utils.config_parser import TrainTest as TrainTestConfig
from src.models.model import Model
from lib.libdataset.src.datasets.dataset import DatasetWrapper
from lib.libdataset.src.utils.tools import Logger
from lib.libdataset.config.consts import General as _CG


class TrainTest(ABC):

    train_str, val_str, test_str = _CG.DEFAULT_SUBSETS
    
    def __init__(self, train_test_config: TrainTestConfig, model: Model, dataset_wrapper: DatasetWrapper, *add_ds):
        self.train_test_config = train_test_config
        self.model = model
        self.dataset_wrapper = dataset_wrapper

        # CHECK to avoid linter's error change abstract property of DatasetWrapper.{split}_dataset to DatasetLauncher
        self.train_info: Optional[dict] = self.dataset_wrapper.train_dataset.info_dict
        self.val_info: Optional[dict] = self.dataset_wrapper.val_dataset.info_dict if self.dataset_wrapper.val_dataset is not None else None
        self.test_info: Optional[dict] = self.dataset_wrapper.test_dataset.info_dict

        self._model_config = self.model.config.model
        self.acc_var: Deque[float] = deque(maxlen=10)

    @abstractmethod
    def train(self):
        ...

    @abstractmethod
    def test(self, model_path: str):
        ...

    def check_stop_conditions(self, loss: float, curr_acc: float, limit: float = 0.985, eps: float = 0.001) -> bool:
        if torch.isnan(torch.tensor(loss)).item():
            Logger.instance().error(f"Raised stop conditions because loss is NaN")
            return True

        if curr_acc < limit:
            return False
        
        if not len(self.acc_var) == self.acc_var.maxlen:
            self.acc_var.append(curr_acc)
            return False
        
        self.acc_var.popleft()
        self.acc_var.append(curr_acc)

        acc_var = torch.Tensor(list(self.acc_var))
        
        if torch.max(acc_var) > 0.999:
            Logger.instance().warning(f"Accuracy is 1.0: hit stop conditions")
            return True
        
        if torch.max(acc_var) - torch.min(acc_var) > 2 * eps:
            return False
        
        Logger.instance().warning(f"Raised stop condition: last {len(self.acc_var)} increment below {2 * eps}")
        return True

    @staticmethod
    def compute_accuracy(y_pred: torch.Tensor, y: torch.Tensor) -> float:
        top_pred = y_pred.argmax(1, keepdim=True)           # select the max class (the one with the highest score)
        correct = top_pred.eq(y.view_as(top_pred)).sum()    # count the number of correct predictions
        return (correct.float() / y.shape[0]).item()        # compute percentage of correct predictions (accuracy score)


class CustomWarmupCosineSchedule(LambdaLR):
  """ Linear warmup and then cosine decay.
      Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
      Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
      If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
  """

  def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1, final_lr=5e-7):
    self.warmup_steps = warmup_steps
    self.t_total = t_total
    self.cycles = cycles
    self.final_lr = final_lr
    super(CustomWarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

  def lr_lambda(self, step):
    if step < self.warmup_steps:
      return float(step) / float(max(1.0, self.warmup_steps))
    # progress after warmup
    progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
    return max(self.final_lr, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
  

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(5e-6, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))