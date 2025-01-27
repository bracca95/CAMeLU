import os
import sys
import wandb
import argparse

from src.utils.main_foo import MainFoo
from src.utils.config_parser import read_from_json
from src.models.model_utils import ModelBuilder
from src.routines.routine_utils import RoutineBuilder
from lib.libdataset.src.datasets.dataset_utils import DatasetBuilder
from lib.libdataset.src.utils.tools import Logger, Tools
from lib.libdataset.config.consts import General as _CG


parser = argparse.ArgumentParser()
parser.add_argument("--config_file", nargs="?", type=str, default=None)
args = vars(parser.parse_args())

def main(config_path: str):
    try:
        config = read_from_json(config_path)
        MainFoo.set_seed(config)
    except Exception as e:
        Logger.instance().critical(e.args)
        sys.exit(-1)

    # load dataset
    try:
        dataset_wrapper = DatasetBuilder.load_dataset(config.dataset)
        config, dw_list = MainFoo.ds_mng(config, config_path, dataset_wrapper)
    except ValueError as ve:
        Logger.instance().critical(ve.args)
        sys.exit(-1)
    
    # get the main dataset and all the others as list (empty if only one is required for training)
    dataset_wrapper = dw_list[0]
    dw_list = [] if len(dw_list) == 1 else dw_list[1:]

    # start main program
    MainFoo.init_wandb(config)

    # instantiate model
    try:
        model = ModelBuilder.load_model(config)
        model = model.to(_CG.DEVICE)
    except ValueError as ve:
        Logger.instance().critical(ve.args)
        sys.exit(-1)

    # train/test
    try:
        routine = RoutineBuilder.build_routine(config.train_test, model, dataset_wrapper, *dw_list)
    except ValueError as ve:
        Logger.instance().critical(ve)
        sys.exit(-1)
        
    if config.train_test.model_test_path is None:
        routine.train()
        model_path = os.path.join(os.getcwd(), "output/best_model.pth")
    
    model_path = config.train_test.model_test_path if config.train_test.model_test_path is not None else model_path
    routine.test(model_path)

    wandb.save(f"{os.path.join(os.getcwd(), 'output', 'log.log')}", base_path=os.getcwd())
    wandb.finish()

if __name__=="__main__":
    config_file_path = args["config_file"] if args["config_file"] is not None else "config/config.json"
    Logger.instance().debug(f"config file located at {config_file_path}")
    main(config_file_path)