import os
import json

def transformer_config():
    path_to_config_transformer = os.path.join('config', 'transformer_config.json')
    config = {
        "num_layers": 4,
        "d_model": 128,
        "dff": 512,
        "num_heads": 8,
        "dropout_rate": 0.1,
        "input_vocab_size": 8500,
        "target_vocab_size": 8000,
        "maximum_position_encoding": 10000,
        "batch_size": 32,
        "tar_seq_len": 16
    }

    os.makedirs(os.path.dirname(path_to_config_transformer), exist_ok=True)

    if not os.path.isfile(path_to_config_transformer):
        with open(path_to_config_transformer, 'w') as f:
            json.dump(config, f)

    try:
        with open(path_to_config_transformer, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        with open(path_to_config_transformer, 'w') as f:
            json.dump(config, f)
    return config


def metrics_config():
    path_to_config_of_train = os.path.join('config', 'metrics_config.json')
    config = {
        "accuracy_set":  'accuracy',
        "mean_sq_error": 'mse',
        "precision": 'precision',
        "thresholds": 0.5
    }

    os.makedirs(os.path.dirname(path_to_config_of_train), exist_ok=True)

    if not os.path.isfile(path_to_config_of_train):
        with open(path_to_config_of_train, 'w') as f:
            json.dump(config, f)

    try:
        with open(path_to_config_of_train, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        with open(path_to_config_of_train, 'w') as f:
            json.dump(config, f)
    return config


