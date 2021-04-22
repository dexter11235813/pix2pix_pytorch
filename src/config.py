from albumentations.pytorch.transforms import ToTensor
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


DEVICE = "cuda"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 1
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True

TRAIN_PATH = "/content/drive/My Drive/pix2pix/data/data/maps/maps/train"
VAL_PATH = "/content/drive/My Drive/pix2pix/data/data/maps/maps/val"

CHECKPOINT_DISC = "/content/drive/My Drive/pix2pix/pix2pix_pytorch/saved_model/disc.pt"
CHECKPOINT_GEN = "/content/drive/My Drive/pix2pix/pix2pix_pytorch/saved_model/gen.pt"

both_transform = A.Compose(
    [A.Resize(width=256, height=256), A.HorizontalFlip(p=0.5)],
    additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.ColorJitter(p=0.1),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

