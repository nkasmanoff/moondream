import os
import matplotlib.pyplot as plt
import torch
import wandb
from matplotlib.patches import Rectangle
import os


def plot_progress_images(epoch, val_dataset, model, use_huggingface):
    if not os.path.exists("progress"):
        os.makedirs("progress")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    for sample_idx, sample in enumerate(val_dataset):
        img_width, img_height = sample["image"].size  # PIL: (width, height)
        if len(sample["class_names"]) == 0 or len(sample["boxes"]) == 0:
            continue

        true_objects = sample["boxes"].cpu().numpy()

        with torch.no_grad():
            if not use_huggingface:
                enc = model.encode_image(sample["image"])
                objects = model.detect(enc, sample["class_names"][0])

            else:
                objects = model.detect(sample["image"], sample["class_names"][0])

        for obj in true_objects:
            x = obj[0] * img_width
            y = obj[1] * img_height
            w = obj[2] * img_width
            h = obj[3] * img_height

            # Draw true bounding box
            rect = Rectangle(
                (x, y),
                w,
                h,
                linewidth=1,
                edgecolor="g",
                facecolor="none",
            )
            ax.add_patch(rect)
            # Add class name text
            ax.text(
                x,
                y,
                sample["class_names"][0],
                color="g",
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1),
            )

        for obj in objects["objects"]:
            x_min = obj["x_min"] * img_width
            y_min = obj["y_min"] * img_height
            x_max = obj["x_max"] * img_width
            y_max = obj["y_max"] * img_height
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            # Draw bounding box
            rect = Rectangle(
                (x_center, y_center),
                width,
                height,
                linewidth=3,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

            # Add class name text
            ax.text(
                x_min,
                y_min,
                sample["class_names"][0],
                color="r",
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1),
            )

        ax.imshow(sample["image"])
        ax.set_axis_off()
        plt.tight_layout()
        filename = f"progress_{epoch}_{sample_idx}.png"
        plt.savefig(os.path.join("progress", filename))
        wandb.log({"progress": wandb.Image(os.path.join("progress", filename))})

        # Clear the axis for the next image
        ax.cla()

        if sample_idx > 15:
            break

    plt.close(fig)
