from PIL import Image
from albumentations import augmentations
import numpy as np
import os
import config
from torch.utils.data import Dataset


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)

        img = np.array(Image.open(img_path))
        input_img = img[:, :600, :]
        trg_img = img[:, 600:, :]
        augmentations = config.both_transform(image=input_img, image0=trg_img)
        input_img, trg_img = augmentations["image"], augmentations["image0"]

        input_img = config.transform_only_input(image=input_img)["image"]
        trg_img = config.transform_only_mask(image=trg_img)["image"]

        return {"input_image": input_img, "target_image": trg_img}


if __name__ == "__main__":
    train_dataset = MapDataset(root_dir=config.TRAIN_PATH)
    print(train_dataset[0]["target_image"].shape)
