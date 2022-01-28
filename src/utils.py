import os
import requests
import tarfile

import hydra


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
    print("Done unzipping.")
