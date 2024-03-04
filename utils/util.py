import os
import pickle

dir_path = "/".join(os.path.dirname(__file__).split("/")[:-1])


def read_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)
    return pkl_data
