# inspired by quicktype.io

from __future__ import annotations

import os
import sys
import json

from functools import reduce
from dataclasses import dataclass, field
from typing import Optional, Union, List, Any, Callable, Iterable, Type, cast

from config.consts import ConfigConst as _CC
from config.consts import ModelConfig as _CM
from config.consts import TrainTestConfig as _CTT
from config.consts import ContextConfig as _CCTX
from lib.libdataset.config.consts import T
from lib.libdataset.config.consts import General as _CG
from lib.libdataset.src.utils.config_parser import DatasetConfig
from lib.libdataset.src.utils.tools import Tools, Logger

def from_bool(x: Any) -> bool:
    Tools.check_instance(x, bool)
    return x

def from_int(x: Any) -> int:
    Tools.check_instance(x, int)
    return x

def from_float(x: Any) -> float:
    Tools.check_instance(x, float)
    return x

def from_str(x: Any) -> str:
    Tools.check_instance(x, str)
    return x

def from_none(x: Any) -> Any:
    Tools.check_instance(x, None)
    return x

def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    Tools.check_instance(x, list)
    return [f(y) for y in x]

def from_union(fs: Iterable[Any], x: Any):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    raise TypeError(f"{x} should be one out of {[type(f.__name__) for f in fs]}")


def to_class(c: Type[T], x: Any) -> dict:
    Tools.check_instance(x, c)
    return cast(Any, x).serialize()


@dataclass
class Context:
    n_layers: int = _CG.DEFAULT_INT
    n_heads: int = _CG.DEFAULT_INT
    mlp_dim: int = _CG.DEFAULT_INT
    hidden_dim: int = _CG.DEFAULT_INT
    attention_dropout: float = float(_CG.DEFAULT_INT)
    n_way: int = _CG.DEFAULT_INT
    k_shot: int = _CG.DEFAULT_INT
    k_query: int = _CG.DEFAULT_INT
    episodes: int = _CG.DEFAULT_INT

    @classmethod
    def deserialize(cls, obj: Any) -> Context:
        try:
            Tools.check_instance(obj, dict)
            n_layers = from_int(obj.get(_CCTX.CONFIG_N_LAYERS))
            n_heads = from_int(obj.get(_CCTX.CONFIG_N_HEADS))
            mlp_dim = from_int(obj.get(_CCTX.CONFIG_MLP_DIM))
            hidden_dim = from_int(obj.get(_CCTX.CONFIG_HIDDEN_DIM))
            attention_dropout = from_float(obj.get(_CCTX.CONFIG_ATTENTION_DROPOUT))
            n_way = from_int(obj.get(_CCTX.CONFIG_N_WAY))
            k_shot = from_int(obj.get(_CCTX.CONFIG_K_SHOT))
            k_query = from_int(obj.get(_CCTX.CONFIG_K_QUERY))
            episodes = from_int(obj.get(_CCTX.CONFIG_EPISODES))
        except TypeError as te:
            Logger.instance().error(
                f"An error occurred while deserializing `context`. " +
                f"Suggested values: n_layers: 12, n_heads: 12, mlp_dim: 3072, hidden_dim: 768 (ViT), 512 (if CLIP)\n",
                te.args
            )
            sys.exit(-1)

        Logger.instance().info(
            f"Context deserialized: n_layers: {n_layers}, n_heads: {n_heads}, mlp_dim: {mlp_dim}, " +
            f"hidden_dim: {hidden_dim}, attention_dropout: {attention_dropout}, n_way: {n_way}, k_shot: {k_shot}, " +
            f"k_query: {k_query}, episodes: {episodes}"
        )

        return Context(n_layers, n_heads, mlp_dim, hidden_dim, attention_dropout, n_way, k_shot, k_query, episodes)
    
    def serialize(self) -> dict:
        result: dict = {}

        result[_CCTX.CONFIG_N_LAYERS] = from_int(self.n_layers)
        result[_CCTX.CONFIG_N_HEADS] = from_int(self.n_heads)
        result[_CCTX.CONFIG_MLP_DIM] = from_int(self.mlp_dim)
        result[_CCTX.CONFIG_HIDDEN_DIM] = from_int(self.hidden_dim)
        result[_CCTX.CONFIG_ATTENTION_DROPOUT] = from_float(self.attention_dropout)
        result[_CCTX.CONFIG_N_WAY] = from_int(self.n_way)
        result[_CCTX.CONFIG_K_SHOT] = from_int(self.k_shot)
        result[_CCTX.CONFIG_K_QUERY] = from_int(self.k_query)
        result[_CCTX.CONFIG_EPISODES] = from_union([from_none, from_int], self.episodes)

        Logger.instance().info(f"Contex serialized {result}")
        return result

@dataclass
class Model:
    model_name: str = _CG.DEFAULT_STR
    freeze: bool = _CG.DEFAULT_BOOL
    pretrained: bool = _CG.DEFAULT_BOOL
    dropout: float = float(_CG.DEFAULT_INT)
    context: Context = field(default_factory=Context)

    @classmethod
    def deserialize(cls, obj: Any) -> Model:
        try:
            Tools.check_instance(obj, dict)
            model_name = from_str(obj.get(_CM.CONFIG_MODEL_NAME))
            freeze = from_bool(obj.get(_CM.CONFIG_FREEZE))
            pretrained = from_bool(obj.get(_CM.CONFIG_PRETRAINED))
            dropout = from_float(obj.get(_CM.CONFIG_DROPOUT))
            context = Context.deserialize(obj.get(_CM.CONFIG_CONTEXT))
        except TypeError as te:
            Logger.instance().error(te.args)
            sys.exit(-1)

        Logger.instance().info(
            f"model_name: {model_name}, freeze: {freeze}, pretrain: {pretrained}, dropout: {dropout}, context: {context}"
        )
        return Model(model_name, freeze, pretrained, dropout, context)
    
    def serialize(self) -> dict:
        result: dict = {}

        result[_CM.CONFIG_MODEL_NAME] = from_str(self.model_name)
        result[_CM.CONFIG_FREEZE] = from_bool(self.freeze)
        result[_CM.CONFIG_PRETRAINED] = from_bool(self.pretrained)
        result[_CM.CONFIG_DROPOUT] = from_float(self.dropout)
        result[_CM.CONFIG_CONTEXT] = to_class(Context, self.context)

        Logger.instance().info(f"Model serialized {result}")
        return result
    

@dataclass
class TrainTest:
    epochs: int = _CG.DEFAULT_INT
    batch_size: int = _CG.DEFAULT_INT
    num_workers: int = _CG.DEFAULT_INT
    model_test_path: Optional[str] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    optimizer: Optional[str] = None

    @classmethod
    def deserialize(cls, obj: Any) -> TrainTest:
        try:
            Tools.check_instance(obj, dict)
            epochs = abs(from_int(obj.get(_CTT.CONFIG_EPOCHS)))
            batch_size = abs(from_int(obj.get(_CTT.CONFIG_BATCH_SIZE)))
            num_workers = abs(from_int(obj.get(_CTT.CONFIG_NUM_WORKERS)))
            model_test_path = from_union([from_none, from_str], obj.get(_CTT.CONFIG_MODEL_TEST_PATH))
            learning_rate = from_union([from_none, from_float], obj.get(_CTT.CONFIG_LEARNING_RATE))
            weight_decay = from_union([from_none, from_float], obj.get(_CTT.CONFIG_WEIGHT_DECAY))
            optimizer = from_union([from_none, from_str], obj.get(_CTT.CONFIG_OPTIMIZER))
        except TypeError as te:
            Logger.instance().error(te.args)
            sys.exit(-1)

        if model_test_path is not None:
            try:
                model_test_path = Tools.validate_path(model_test_path)
            except FileNotFoundError as fnf:
                msg = f"Check the test path again"
                Logger.instance().critical(f"{fnf.args}.\n{msg}")
                sys.exit(-1)

        Logger.instance().info(
            f"epochs: {epochs}, batch_size: {batch_size}, num_workers: {num_workers}, " +
            f"model_test_path: {model_test_path},learning_rate: {learning_rate}, weight_decay {weight_decay}, " +
            f"optimizer: {optimizer}"
        )
        
        return TrainTest(epochs, batch_size, num_workers, model_test_path, learning_rate, weight_decay, optimizer)
    
    def serialize(self) -> dict:
        result: dict = {}

        result[_CTT.CONFIG_EPOCHS] = from_int(self.epochs)
        result[_CTT.CONFIG_BATCH_SIZE] = from_int(self.batch_size)
        result[_CTT.CONFIG_NUM_WORKERS] = from_int(self.num_workers)
        result[_CTT.CONFIG_MODEL_TEST_PATH] = from_union([from_none, from_str], self.model_test_path)
        result[_CTT.CONFIG_LEARNING_RATE] = from_union([from_none, from_float], self.learning_rate)
        result[_CTT.CONFIG_WEIGHT_DECAY] = from_union([from_none, from_float], self.weight_decay)
        result[_CTT.CONFIG_OPTIMIZER] = from_union([from_none, from_str], self.optimizer)

        Logger.instance().info(f"TrainTest serialized: {result}")
        return result


@dataclass
class Config:
    experiment_name: str = "random_generated_name"
    seed: int = _CG.DEFAULT_INT
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: Model = field(default_factory=Model)
    train_test: TrainTest = field(default_factory=TrainTest)

    @classmethod
    def deserialize(cls, obj: Any) -> Config:
        try:
            Tools.check_instance(obj, dict)
            experiment_name = from_str(obj.get(_CC.CONFIG_EXPERIMENT_NAME))
            seed = from_int(obj.get(_CC.CONFIG_SEED))
            dataset = DatasetConfig.deserialize(obj.get(_CC.CONFIG_DATASET))
            model = Model.deserialize(obj.get(_CC.CONFIG_MODEL))
            train_test = TrainTest.deserialize(obj.get(_CC.CONFIG_TRAIN_TEST))
        except TypeError as te:
            Logger.instance().critical(te.args)
            sys.exit(-1)

        # for FSL/ICL the aug_times must match k_shot
        if dataset.augment_times is not None and dataset.augment_times != model.context.k_shot:
            raise ValueError(
                f"`dataset.augument_times` ({dataset.augment_times}) must match " +
                f"`model.context.k_shot` ({model.context.k_shot}) when the first is not None"
            )
        
        # also cannot be None if augment online is not None
        if dataset.augment_online is not None and dataset.augment_times is None:
            raise ValueError(f"`dataset.augment_times` cannot be None if `dataset.augment_online` is not None")
        
        Logger.instance().info(
            f"Config deserialized: experiment_name: {experiment_name}, seed: {seed}, dataset: {dataset}, " +
            f"model: {model}, train_test: {train_test}"
        )
        
        return Config(experiment_name, seed, dataset, model, train_test)
    
    def serialize(self) -> dict:
        result: dict = {}
        
        # if you do not want to write null values, add a field to result if and only if self.field is not None
        result[_CC.CONFIG_EXPERIMENT_NAME] = from_str(self.experiment_name)
        result[_CC.CONFIG_SEED] = from_int(self.seed)
        result[_CC.CONFIG_DATASET] = to_class(DatasetConfig, self.dataset)
        result[_CC.CONFIG_MODEL] = to_class(Model, self.model)
        result[_CC.CONFIG_TRAIN_TEST] = to_class(TrainTest, self.train_test)

        Logger.instance().info(f"Config serialized {result}")
        return result


def read_from_json(str_path: str) -> Config:
    obj = Tools.read_json(str_path)
    return Config.deserialize(obj)

def write_to_json(config: Config, directory: str, filename: str) -> None:
    dire = None
    
    try:
        dire = Tools.validate_path(directory)
    except FileNotFoundError as fnf:
        Logger.instance().critical(f"{fnf.args}")
        sys.exit(-1)

    serialized_config = config.serialize()
    with open(os.path.join(dire, filename), "w") as f:
        json_dict = json.dumps(serialized_config, indent=4)
        f.write(json_dict)
