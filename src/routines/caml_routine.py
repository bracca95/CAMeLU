import os
import sys
import torch
import wandb

from tqdm import tqdm
from typing import Optional, Union, Tuple, List, Iterator
from functools import reduce
from dataclasses import dataclass, field
from torch.utils.data import DataLoader

from src.models.model import Model
from src.routines.routine import TrainTest, CustomWarmupCosineSchedule
from src.samplers.context_sampler import CtxBatchSampler
from src.utils.config_parser import TrainTest as TrainTestConfig
from lib.libdataset.src.datasets.dataset import DatasetWrapper
from lib.libdataset.src.utils.tools import Logger, Tools
from lib.libdataset.config.consts import General as _CG


def subclass_must_implement(method):
    def wrapper(self, *args, **kwargs):
        if method.__name__ not in dir(self.__class__):
            raise NotImplementedError("Subclasses must implement this method.")
        return method(self, *args, **kwargs)
    return wrapper


@dataclass
class Trainloaders:
    trainloader: Optional[Union[DataLoader, List[DataLoader]]] = None
    supports: List[DataLoader] = field(default_factory=list)
    queries: List[DataLoader] = field(default_factory=list)

    def __len__(self) -> int:
        if self.trainloader is None:
            return len(self.supports)
        if type(self.trainloader) is list:
            return len(self.trainloader)
        return 1

    def get_all_len(self):
        # return the cumulative length of all the samples across all the datasets
        if self.trainloader is not None:
            if type(self.trainloader) is not list:
                return [len(self.trainloader.dataset)]
            loads = self.trainloader
        else:
            loads = self.supports

        lengths = [len(l.dataset) for l in loads]
        return [reduce(lambda x, y: x + y, lengths[:i+1]) for i in range(len(lengths))]


class CamlRoutine(TrainTest):

    def __init__(self, train_test_config: TrainTestConfig, model: Model, dataset_wrapper: DatasetWrapper):
        super().__init__(train_test_config, model, dataset_wrapper)
        
        self.episodes = self._model_config.context.episodes
        if self.episodes is None:
            Logger.instance().warning(f"Overriding None value for 'episodes'. Defaulting to 200")
            self.episodes = 2 * (self.train_test_config.epochs)
        
        if self.train_test_config.learning_rate is None:
            Logger.instance().warning(f"learning rate has not been set in config. Defaulting to 0.001")
            self.lr = 0.001
        else:
            self.lr = self.train_test_config.learning_rate

        if self.train_test_config.weight_decay is None:
            Logger.instance().warning(f"weight decay has not been set in config. Defaulting to 0.03")
            self.wdecay = 0.03
        else:
            self.wdecay = self.train_test_config.weight_decay

    @subclass_must_implement
    def init_trainloaders(self) -> Trainloaders:
        return Trainloaders(self.init_loader(self.train_str))
    
    @subclass_must_implement
    def get_iters(self, trainloaders: Trainloaders, epoch: int) -> Tuple[List[Iterator], Optional[List[Iterator]]]:
        return [iter(trainloaders.trainloader)], None
    
    @subclass_must_implement
    def choose_task_dataset(
        self,
        iter_list_1: List[Iterator],
        iter_list_2: Optional[List[Iterator]],
        dataset_lengths: List[int]
    ) -> Tuple[Iterator, Optional[Iterator]]:
        return iter_list_1[0], None

    @subclass_must_implement
    def get_img_lbl_from_loaders(self, loader_1: Iterator, loader_2: Optional[Iterator]):
        x, y = next(loader_1)
        return x, y
    
    @subclass_must_implement
    def init_loader(self, split_set: str, augment: Optional[str]=None) -> Optional[DataLoader]:
        train_set, val_set, test_set = _CG.DEFAULT_SUBSETS
        current_dataset = getattr(self.dataset_wrapper, f"{split_set}_dataset")
        
        if current_dataset is None:
            return None

        sampler = CtxBatchSampler(
            labels=current_dataset.label_list,
            classes_per_it=self._model_config.context.n_way,
            n_samples_cls=self._model_config.context.k_shot + self._model_config.context.k_query,
            iterations=self.episodes
        )
    
        return DataLoader(current_dataset, batch_sampler=sampler, num_workers=self.train_test_config.num_workers)

    def train(self):
        Logger.instance().debug("Training called: loading dataloaders...")

        trainloaders = self.init_trainloaders()
        valloader = self.init_loader(self.val_str)
        
        dataset_lengths = trainloaders.get_all_len()

        optim_param = [{"params": self.model.parameters()}]
        optim = torch.optim.Adam(params=optim_param, lr=self.lr, weight_decay=self.wdecay)
        scheduler = CustomWarmupCosineSchedule(
            optim,
            warmup_steps=3 * self.episodes,
            t_total=self.train_test_config.epochs * self.episodes,
            final_lr=self.lr / 10
        )

        criterion = torch.nn.CrossEntropyLoss()

        best_acc: float = 0.0
        best_acc_val: float = 0.0
        best_loss = float("inf")

        # create output folder to store data
        out_folder = os.path.join(os.getcwd(), "output")
        os.makedirs(out_folder, exist_ok=True)

        best_model_path = os.path.join(out_folder, "best_model.pth")

        ctx_cfg = self._model_config.context
        n_way, k_shot, k_query = (ctx_cfg.n_way, ctx_cfg.k_shot, ctx_cfg.k_query)
        
        Logger.instance().debug("Start training!")
        for epoch in range(self.train_test_config.epochs):
            Logger.instance().debug(f"=== Epoch: {epoch} ===")
            self.model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0

            loader_list_1, loader_list_2 = self.get_iters(trainloaders, epoch)
            
            for episode in tqdm(range(self.episodes)):
                optim.zero_grad()
                loader_1, loader_2 = self.choose_task_dataset(loader_list_1, loader_list_2, dataset_lengths)
                x, y = self.get_img_lbl_from_loaders(loader_1, loader_2)
                
                x = x.to(_CG.DEVICE)
                y = y.to(_CG.DEVICE)
                
                # prediction
                preds, yq = self.model(x, y)
                loss = criterion(preds, yq)
                acc = self.compute_accuracy(preds, yq)

                # backward pass & scheduler
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optim.step()
                scheduler.step()

                # update values
                epoch_loss += loss.item()
                epoch_acc += acc

            epoch_loss = epoch_loss / self.episodes
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                Logger.instance().debug(f"Found best loss at epoch {epoch}. Loss = {epoch_loss:.5f}")

            epoch_acc = epoch_acc / self.episodes
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                Logger.instance().debug(f"Found best accuracy at epoch {epoch}. Accuracy: {epoch_acc:.5f}")

            # wandb
            wdb_dict = { "train_loss": epoch_loss, "train_acc": epoch_acc }

            ## VALIDATION
            if valloader is not None:
                epoch_acc_val = self.validate(valloader)
                if epoch_acc_val >= best_acc_val:
                    best_acc_val = epoch_acc_val
                    Logger.instance().debug(f"Found the best evaluation model at epoch {epoch}!")
                    torch.save(self.model.state_dict(), best_model_path)

                # wandb
                wdb_dict["val_acc"] = epoch_acc_val    
            ## EOF: VALIDATION
            
            # wandb
            wandb.log(wdb_dict)

            # stop conditions and save last model
            if epoch == self.train_test_config.epochs-1:
                Logger.instance().debug(f"=== STOP ===")

                # wandb: save all models
                wandb.save(f"{out_folder}/*.pth", base_path=os.getcwd())

                return

    def validate(self, valloader: DataLoader):
        Logger.instance().debug(f"= Validating! =")
        self.model.eval()
        with torch.no_grad():
            epoch_acc = 0.0
            for x, y in tqdm(valloader, total=len(valloader)):
                x = x.to(_CG.DEVICE)
                y = y.to(_CG.DEVICE)
                
                # prediction
                preds, yq = self.model(x, y)
                acc = self.compute_accuracy(preds, yq)

                # update values
                epoch_acc += acc

            epoch_acc = epoch_acc / len(valloader)
            Logger.instance().debug(f"Validation accuracy: {epoch_acc}")

            return epoch_acc

    def test(self, model_path: str):
        Logger.instance().debug("Start testing")
        
        try:
            model_path = Tools.validate_path(model_path)
            testloader = self.init_loader(self.test_str)
        except FileNotFoundError as fnf:
            Logger.instance().critical(f"model not found: {fnf.args}")
            sys.exit(-1)
        except ValueError as ve:
            Logger.instance().error(f"{ve.args}. No test performed")
            return
        
        best_acc: float = 0.0
        mean_acc: float = 0.0
        n_it_test = 10
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        with torch.no_grad():
            for epoch in tqdm(range(n_it_test), total=n_it_test):
                epoch_acc = 0.0
                for x, y in testloader:
                    x = x.to(_CG.DEVICE)
                    y = y.to(_CG.DEVICE)
                    
                    # prediction
                    preds, yq = self.model(x, y)
                    acc = self.compute_accuracy(preds, yq)

                    # update values
                    epoch_acc += acc

                epoch_acc = epoch_acc / len(testloader)
                mean_acc += epoch_acc
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
            
            mean_acc = mean_acc / n_it_test

        Logger.instance().debug(f"Test accuracy: {best_acc:.5f}, on avg: {mean_acc:.5f}")
