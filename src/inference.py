import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection import backbone_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from tqdm import tqdm
import json

from dataset import MyDataset, collate_fn
import utils

# Instantiate dataset and dataloaders
dataset = MyDataset("data/test_release", isTrain=False)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    collate_fn=collate_fn
)

backbone = backbone_utils.resnet_fpn_backbone('resnet101', pretrained=True)
# anchor generator
anchor_sizes = ((16,), (32,), (48,), (64,), (96,))
aspect_ratios = ((0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0),) \
    * len(anchor_sizes)
anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
print(anchor_generator.num_anchors_per_location())

model = MaskRCNN(
    backbone=backbone,
    num_classes=5,
    rpn_anchor_generator=anchor_generator
)


# 5. Modify the model to match the number of classes in your dataset
# (e.g., 4 classes including background)
num_classes = 5  # 4 classes + 1 background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = \
    torchvision.models.detection \
    .faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load("checkpoint/maskrcnn_epoch_150.pth"))

# 6. Set the model to training mode
model.eval()
device = "cuda"

model = model.to(device)


pbar = tqdm(dataloader)
predictions = []
with torch.no_grad():
    for batch in pbar:
        images = batch["images"]
        print(len(images))
        original_res = batch["targets"][0]["original_res"]
        images = [image.to(device) for image in images]
        prediction = model(images)  # boxes, labels, scores, masks
        print("boxes", prediction[0]["boxes"].shape)
        print("labels", prediction[0]["labels"].shape)
        print("scores", prediction[0]["scores"].shape)
        print("masks", prediction[0]["masks"].shape)
        print("masks", prediction[0]["masks"].dtype)
        print("origianl_res", original_res)

        scale_y = original_res[0] / 1024
        scale_x = original_res[1] / 1024

        prediction[0]["boxes"][:, 0] *= scale_x
        prediction[0]["boxes"][:, 2] *= scale_x
        prediction[0]["boxes"][:, 1] *= scale_y
        prediction[0]["boxes"][:, 3] *= scale_y

        # change the output from the [xmin, ymin, xmax, ymax]
        # to [xmin, ymin, width, height]
        prediction[0]["boxes"][:, 2] -= prediction[0]["boxes"][:, 0]
        prediction[0]["boxes"][:, 3] -= prediction[0]["boxes"][:, 1]

        prediction[0]["masks"] = prediction[0]["masks"].to("cpu")
        mask_resized = np.zeros(
            (prediction[0]["masks"].shape[0], 1, *original_res)
        )
        mask_resized = transforms.Resize(
            original_res,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )(prediction[0]["masks"])
        print(mask_resized.shape)

        for i in range(prediction[0]["masks"].shape[0]):
            output = {
                'image_id': dataset.filenameToId(
                    batch["targets"][0]["image_name"]
                ),
                'bbox': prediction[0]["boxes"][i, :].cpu().numpy().tolist(),
                'score': float(prediction[0]["scores"][i].cpu().numpy()),
                'category_id': int(prediction[0]["labels"][i].cpu().numpy()),
                'segmentation': utils.encode_mask(
                    mask_resized[i, 0, :, :] > 0.5
                )
            }
            print(
                output
            )
            predictions.append(
                output
            )

    with open("test-results.json", "w") as f:
        json.dump(predictions, f, indent=4)
