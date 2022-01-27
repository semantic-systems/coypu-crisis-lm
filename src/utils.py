import os
import hydra


def generate_data_paths(cfg):
    data_folder = hydra.utils.to_absolute_path(cfg.data_path)
    data_paths = {}
    modes = ["train", "dev", "test"]
    for f in os.listdir(data_folder):
        for mode in modes:
            if all(x in f for x in [cfg.task, mode]):
                data_paths[mode] = os.path.join(data_folder, f)
    return data_paths