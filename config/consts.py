# about TypedDict https://stackoverflow.com/a/64938100


class ConfigConst:
    CONFIG_EXPERIMENT_NAME = "experiment_name"
    CONFIG_SEED = "seed"
    CONFIG_DATASET = "dataset"
    CONFIG_MODEL = "model"
    CONFIG_TRAIN_TEST = "train_test"

class ModelConfig:
    CONFIG_MODEL_NAME = "model_name"
    CONFIG_FREEZE = "freeze"
    CONFIG_PRETRAINED = "pretrained"
    CONFIG_DROPOUT = "dropout"
    CONFIG_CONTEXT = "context"

class ContextConfig:
    CONFIG_N_LAYERS = "n_layers"
    CONFIG_N_HEADS = "n_heads"
    CONFIG_MLP_DIM = "mlp_dim"
    CONFIG_HIDDEN_DIM = "hidden_dim"
    CONFIG_ATTENTION_DROPOUT = "attention_dropout"
    CONFIG_N_WAY = "n_way"
    CONFIG_K_SHOT = "k_shot"
    CONFIG_K_QUERY = "k_query"
    CONFIG_EPISODES = "episodes"

class TrainTestConfig:
    CONFIG_EPOCHS = "epochs"
    CONFIG_BATCH_SIZE = "batch_size"
    CONFIG_NUM_WORKERS = "num_workers"
    CONFIG_MODEL_TEST_PATH = "model_test_path"
    CONFIG_LEARNING_RATE = "learning_rate"
    CONFIG_WEIGHT_DECAY = "weight_decay"
    CONFIG_OPTIMIZER = "optimizer"
