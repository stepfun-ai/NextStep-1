import io
import json

from nextstep.utils.image_utils import to_rgb


def npy_loads(data):
    """Load data from npy format. Imports numpy only if necessary."""
    import numpy.lib.format

    stream = io.BytesIO(data)
    return numpy.lib.format.read_array(stream)


def npz_loads(data):
    """Load data from npz format. Imports numpy only if necessary."""
    import numpy as np

    stream = io.BytesIO(data)
    return dict(np.load(stream))


def img_loads(data):
    import PIL.Image
    import PIL.ImageOps

    image = PIL.Image.open(io.BytesIO(data))
    image = to_rgb(image)
    return image


def pickle_loads(data):
    import pickle

    return pickle.loads(data)


# fmt: off
default_handlers = {
    ".txt"   : lambda data : data.decode("utf-8"),
    ".json"  : lambda data : json.loads(data),
    ".npy"   : npy_loads,
    ".npz"   : npz_loads,
    ".jpg"   : img_loads,
    ".jpeg"  : img_loads,
    ".png"   : img_loads,
    ".webp"  : img_loads,
    ".pickle": pickle_loads,
    ".mp4"   : lambda data : data,
}
# fmt: on
