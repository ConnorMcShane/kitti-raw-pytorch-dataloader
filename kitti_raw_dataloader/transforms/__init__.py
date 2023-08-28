"""Init file"""
from .normalise import Normalise
from .resize import Resize
from .random_flip import RandomFlip
from .to_tensor import ToTensor
from .crop import Crop
from .random_crop import RandomCrop

TRANSFORMS = {
    "Normalise": Normalise,
    "Resize": Resize,
    "RandomFlip": RandomFlip,
    "ToTensor": ToTensor,
    "Crop": Crop,
    "RandomCrop": RandomCrop,
}
