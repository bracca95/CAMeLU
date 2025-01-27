from typing import Union
from torch.utils.data import Dataset

from src.models.model import Model
from src.routines.routine import TrainTest
from src.routines.caml_routine import CamlRoutine
from src.routines.caml_routine_multi import CamlRoutineMulti
from src.routines.camlu_routine_strong import CamluStrongRoutine
from src.routines.camlu_routine_mixup import CamluMixupRoutine
from src.routines.camlu_routine_mixup_multi import CamluMixupMultiRoutine
from src.utils.config_parser import TrainTest as TrainTestConfig
from lib.libdataset.src.utils.tools import Logger
from lib.libdataset.src.datasets.dataset import DatasetWrapper

class RoutineBuilder:

    @staticmethod
    def build_routine(
        train_test_config: TrainTestConfig,
        model: Model,
        dataset_wrapper: Union[DatasetWrapper, Dataset],
        *args
    ) -> TrainTest:
        
        # ----------------------------------------- CAML -----------------------------------------------
        if model.config.dataset.augment_online is None:
            # CAML multi-dataset
            if model.config.dataset.dataset_id is not None and len(model.config.dataset.dataset_id) > 1:
                return CamlRoutineMulti(train_test_config, model, dataset_wrapper, *args)
            
            # standard CAML
            return CamlRoutine(train_test_config, model, dataset_wrapper)
        
        # --------------------------------------- CAMeLU ---------------------------------------------
        if set(model.config.dataset.augment_online) == set(["support", "query"]):
            # CAMeLU multi-dataset
            if model.config.dataset.dataset_id is not None and len(model.config.dataset.dataset_id) > 1:
                return CamluMixupMultiRoutine(train_test_config, model, dataset_wrapper, *args)
            
            # standard CAMeLU
            return CamluMixupRoutine(train_test_config, model, dataset_wrapper)

        # CAMeLU ablation: strong aumentation
        if set(model.config.dataset.augment_online) == set(["dataset"]):
            return CamluStrongRoutine(train_test_config, model, dataset_wrapper)
        
        raise ValueError(f"Accepted values for `augment_online`: `['dataset']`, `['support', 'query']`")