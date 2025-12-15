import os
import pickle
from torchvision import datasets

# Directory structure:
# <this_file>/../data/mnist/
MODULE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "mnist")
)

def download_mnist():
    os.makedirs(MODULE_DIR, exist_ok=True)

    # Torchvision will create MNIST/{raw,processed} inside this dir
    train_dataset = datasets.MNIST(
        root=MODULE_DIR, train=True, download=True
    )
    test_dataset = datasets.MNIST(
        root=MODULE_DIR, train=False, download=True
    )

    # Convert to NumPy and flatten images
    train_images = train_dataset.data.numpy().reshape(-1, 28 * 28)
    train_labels = train_dataset.targets.numpy()
    test_images = test_dataset.data.numpy().reshape(-1, 28 * 28)
    test_labels = test_dataset.targets.numpy()

    mnist = {
        "training_images": train_images,
        "training_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }

    pkl_path = os.path.join(MODULE_DIR, "mnist.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(mnist, f)

    print(f"MNIST saved to {pkl_path}")


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


if __name__ == "__main__":
    download_mnist()
