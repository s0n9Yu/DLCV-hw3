import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torchvision
import skimage.io as sio
import cv2
from PIL import Image
import torchvision.transforms.functional
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

from collections import defaultdict


class MyDataset(Dataset):
    def __init__(self, directory, isTrain=True):
        self.rootdir = directory
        self.imagedir = os.listdir(directory)
        self.isTrain = isTrain
        if not self.isTrain:
            dict_path = os.path.join(self.rootdir,
                                     "..",
                                     "test_image_name_to_ids.json")
            with open(dict_path, "r") as file:
                self.iddict = json.load(file)
            print(self.iddict)

    def __len__(self):
        return len(self.imagedir)

    def __getitem__(self, idx):
        if self.isTrain:
            image_path = os.path.join(
                self.rootdir,
                self.imagedir[idx],
                "image.tif"
            )
            image = Image.open(image_path).convert("RGB")
            image = torchvision.transforms.functional \
                .pil_to_tensor(image) \
                .to(torch.float)
            image = image / 255.0

            res = (1024, 1024)
            newHeight, newWidth = res

            image = F.resize(image, res)

            bbox = []
            try:
                class1 = sio.imread(
                    os.path.join(
                        self.rootdir,
                        self.imagedir[idx],
                        "class1.tif"
                    )
                )
                bbox = bbox + self.maskToBoundingBox(class1, 1, idx, res)
            except FileNotFoundError:
                class1 = None
            try:
                class2 = sio.imread(
                    os.path.join(
                        self.rootdir,
                        self.imagedir[idx],
                        "class2.tif"
                    )
                )
                bbox = bbox + self.maskToBoundingBox(class2, 2, idx, res)
            except FileNotFoundError:
                class2 = None
            try:
                class3 = sio.imread(
                    os.path.join(
                        self.rootdir,
                        self.imagedir[idx],
                        "class3.tif"
                    )
                )
                bbox = bbox + self.maskToBoundingBox(class3, 3, idx, res)
            except FileNotFoundError:
                class3 = None
            try:
                class4 = sio.imread(
                    os.path.join(
                        self.rootdir,
                        self.imagedir[idx],
                        "class4.tif"
                    )
                )
                bbox = bbox + self.maskToBoundingBox(class4, 4, idx, res)
            except FileNotFoundError:
                class4 = None

            # Merge
            merged = defaultdict(list)
            for d in bbox:
                for k, v in d.items():
                    merged[k].append(v)

            # Concatenate
            for k in merged:
                merged[k] = torch.cat(merged[k], dim=0)

            return image, merged
        else:
            image_path = os.path.join(self.rootdir, self.imagedir[idx])
            image = Image.open(image_path).convert("RGB")
            image = torchvision.transforms.functional \
                .pil_to_tensor(image).to(torch.float)
            image = image / 255.0

            res_original = image.shape[1:]
            res = (1024, 1024)
            newHeight, newWidth = res

            image = F.resize(image, res)

            return image, \
                {
                    "original_res": res_original,
                    "image_name": self.imagedir[idx]
                }

    def filenameToId(self, filename):
        tmp = [t for t in self.iddict if t["file_name"] == filename]
        assert len(tmp) == 1, "filename not found"
        return tmp[0]["id"]

    @staticmethod
    def maskToBoundingBox(mask, classID, imageID, resolution):
        if not isinstance(mask, np.ndarray):
            return []

        allInstance = np.unique(mask)[1:]
        bboxs = []

        for instanceId in allInstance:
            instanceMask = (mask == instanceId)
            # Assuming 'mask' is a boolean numpy array (dtype: bool)
            # Convert boolean to uint8 (0, 1)
            mask_uint8 = instanceMask.astype(np.uint8)
            # Resize using nearest neighbor interpolation
            resized_mask = cv2.resize(
                mask_uint8,
                resolution,
                interpolation=cv2.INTER_NEAREST
            )
            # Convert back to boolean if desired
            instanceMask = resized_mask.astype(np.bool)

            rows = np.any(instanceMask, axis=0)
            cols = np.any(instanceMask, axis=1)

            xmin, xmax = np.where(rows)[0][[0, -1]]
            ymin, ymax = np.where(cols)[0][[0, -1]]

            area = np.count_nonzero(instanceMask)

            bboxs.append({
                "labels": torch.tensor([classID], dtype=torch.int64),
                "masks": torch.tensor(instanceMask).unsqueeze(0),
                "boxes": torch.from_numpy(
                    np.array([[xmin, ymin, xmax, ymax]])
                ),
                "image_id": torch.tensor([imageID], dtype=torch.int),
                "area": torch.tensor([area]),
                "iscrowd": torch.tensor([0])
            })
        return bboxs


def collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]

    return {"images": images, "targets": targets}


if __name__ == "__main__":
    print("-----train----")
    dataset = MyDataset("data/train")
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn
    )

    batch = next(iter(train_loader))
    print(len(batch['images']))
    print(len(batch['targets']))
    for k, v in batch['targets'][0].items():
        print(k, v.shape, v.dtype)
    print("image", batch["images"][0].shape)
    print("image", batch["images"][0].max())
    print("image", batch["images"][0].min())
    color = ['r', 'r', 'g', 'b', 'y']
    image = batch['images'][0].permute(1, 2, 0).numpy()
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    colored_mask = np.zeros((1024, 1024, 3))
    for box, label, mask in zip(
            batch['targets'][0]['boxes'],
            batch['targets'][0]['labels'],
            batch['targets'][0]['masks']
            ):
        xmin, ymin, xmax, ymax = box.tolist()
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor='b',
            facecolor='none'
        )
        ax.add_patch(rect)
        color = [1, 1, 1]
        for c in range(3):
            colored_mask[:, :, c] = colored_mask[:, :, c] \
                + mask.numpy() * color[c]

    ax.imshow(colored_mask, alpha=0.5)
    plt.axis("off")
    plt.savefig("test.png")

    exit(0)
    print("-----test----")
    test_dataset = MyDataset("data/test_release", isTrain=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn
    )

    batch = next(iter(test_loader))
    print(len(batch['images']))
    print(len(batch['targets']))
