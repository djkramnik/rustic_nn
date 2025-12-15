import os

import pickle

MODULE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "mnist")
)

def load():
    pkl_path = os.path.join(MODULE_DIR, "mnist.pkl")
    with open(pkl_path, "rb") as f:
        mnist = pickle.load(f)

    return (
        mnist["training_images"],
        mnist["training_labels"],
        mnist["test_images"],
        mnist["test_labels"],
    )