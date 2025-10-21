import os

DATASET_ROOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets"
)

from .base import DataLoaderBase, FileTypeNotSupportedError
from .davis_data import DavisDataLoader
from .dnd21 import Dnd21DataLoader
from .end import EndDataLoader

# List of supported dataset
def inheritors(klass):
    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


collections = {k.NAME: k for k in inheritors(DataLoaderBase)}
collections.update({"DAVIS": DavisDataLoader})  # Other name cases


def setup(config: dict) -> DataLoaderBase:
    loader = collections[config["dataset"]](config=config)
    loader.set_sequence(config["sequence"])
    return loader
