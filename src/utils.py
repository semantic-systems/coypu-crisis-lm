import os
from pathlib import Path
import requests
import tarfile

import hydra


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def download_data_from_url(cfg):
    dest_path = hydra.utils.to_absolute_path(cfg.data_path)
    os.makedirs(dest_path, exist_ok=True)
    dest_file = os.path.join(dest_path, os.path.basename(cfg.download_url))
    if not os.path.isfile(dest_file):
        print("Downloading file", cfg.download_url)
        r = requests.get(cfg.download_url, allow_redirects=True)

        open(dest_file, 'wb').write(r.content)
        print("Done downloading. Stored", dest_file)
    return dest_file


def unzip_tar_file(filename):
    print("Unzipping", filename)
    out_path = hydra.utils.to_absolute_path(os.path.dirname(filename))
    if filename.endswith("tar.gz"):
        tar = tarfile.open(filename, "r:gz")
        tar.extractall(path=out_path)
        tar.close()
    elif filename.endswith("tar"):
        tar = tarfile.open(filename, "r:")
        tar.extractall(path=out_path)
        tar.close()
    os.remove(filename)
    print("Done unzipping.")
    os.remove()


def get_current_artifacts_dir(cfg):
    # Get artifacts dir of current mlflow run
    mlflow_dir = os.path.join(hydra.utils.to_absolute_path(cfg.mlruns_dir), "0")
    latest_run_dir = max([f for f in os.listdir(mlflow_dir)], key=lambda x: os.stat(os.path.join(
        mlflow_dir, x)).st_mtime)
    output_dir = os.path.join(mlflow_dir, latest_run_dir, "artifacts")
    return output_dir
