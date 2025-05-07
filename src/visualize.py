import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np

def visualize_bbox(json_path, image_id, image):
    # Load annotations
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Filter for the specified image_id
    bboxes = [d for d in data if d["image_id"] == image_id]

    # Show the image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    image_np = np.array(image.convert("RGB").getdata())
    print("image_np", image_np.shape)
    #mask_visualize = np.zeros(image_np.shape[:-1])

    # Draw all bounding boxes for this image_id
    for bbox_data in bboxes:
        x, y, width, height = bbox_data["bbox"]
        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)

        #mask = bbox_data["segmentation"]["counts"]
        #mask = utils.decode_maskobj(mask)
        #mask_visualize += mask

        # Optional: add category_id and score
        cat = bbox_data["category_id"]
        score = bbox_data["score"]
        ax.text(x, y - 5, f"cat:{cat}, score:{score:.2f}", color='white', 
                fontsize=8, backgroundcolor='red')
    
    #ax.imshow(0.3 * mask_visualize)

    plt.axis('off')
    plt.savefig("visualize.png")



with open(os.path.join("data/test_release", ".." ,"test_image_name_to_ids.json"), "r") as file:
    iddict = json.load(file)

ID = 17
tmp = [t for t in iddict if t["id"] == ID]
assert len(tmp) == 1, "filename not found"
imgname = tmp[0]["file_name"]
print(imgname)

from PIL import Image

img_path = os.path.join("data/test_release", imgname)
print(img_path)
image = Image.open(img_path)

visualize_bbox('test-results.json', image_id=12, image=image)