import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection import backbone_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import MyDataset, collate_fn

# Instantiate dataset and dataloaders
dataset = MyDataset("data/train", isTrain=True)
dataloader = DataLoader(
    dataset,
    batch_size=2,
    num_workers=4,
    shuffle=True,
    collate_fn=collate_fn
)

# 4. Load the pre-trained Mask R-CNN model
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

# 6. Set the model to training mode
model.train()

# 7. Set up the optimizer (we will use SGD for this example)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.00005)
device = "cuda"

# 8. Define the learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=50,
    gamma=0.8
)

model = model.to(device)

loss_list = []

# 9. Training loop
num_epochs = 150
for epoch in range(num_epochs):
    print(f"epoch: {epoch + 1}")
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader)
    for batch in pbar:
        images = batch["images"]
        targets = batch["targets"]
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)

        # Total loss
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimization
        losses.backward()
        optimizer.step()

        # Track the loss
        running_loss += losses.item()

    # Step the scheduler
    lr_scheduler.step()

    loss_list.append(running_loss / len(dataloader))

    # Print statistics
    print(f"Epoch #{epoch + 1} Loss: {running_loss / len(dataloader)}")

    # Optionally, save model checkpoints
    if (epoch + 1) % 5 == 0:
        torch.save(
            model.state_dict(),
            f"checkpoint/maskrcnn_epoch_{epoch + 1}.pth"
        )

print("Training completed!")

plt.plot(loss_list)
plt.savefig("traincurve.png")
