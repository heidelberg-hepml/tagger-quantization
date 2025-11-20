import numpy as np
import h5py
import hdf5plugin
import os, sys
import wget

# dataset sizes: toptagging 1.5G, event-generation 4.7G
BASE_URL = "https://www.thphys.uni-heidelberg.de/~plehn/data"
FILENAMES = {
    "toptagging": "toptagging_full.npz",
}
DATA_DIR = "data"


def load(filename):
    url = os.path.join(BASE_URL, filename)
    print(f"Started to download {url}")
    target_path = os.path.join(DATA_DIR, filename)
    wget.download(url, out=target_path)
    print("")
    print(f"Successfully downloaded {target_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <command>")
        sys.exit(1)
    dataset = sys.argv[1]

    # collect toptagging dataset
    # this is a npz version of the original dataset at https://zenodo.org/records/2603256
    filename = FILENAMES["toptagging"]
    if dataset == "toptagging":
        load(filename)

if __name__ == "__main__":
    main()
