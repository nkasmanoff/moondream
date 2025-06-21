"""
Roadmap for updates:

- Add image augmentation / label augmentation - x
- Add a validation and test set
- Fine tune on vision encoder - x
- Look into why saving the model does not work
- Try to implement batch processing with dataloaders
- Get more data / use other sources
- Combined datasets?
"""

import random
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import math
from safetensors.torch import save_file
import datasets
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm
from bitsandbytes.optim import AdamW
import wandb
import os
import argparse
from contextlib import nullcontext

from ..torch.weights import load_weights_into_model
from ..torch.moondream import MoondreamModel, MoondreamConfig, text_encoder
from ..torch.text import _produce_hidden
from ..torch.region import (
    decode_coordinate,
    decode_size,
    encode_coordinate,
    encode_size,
)


# This is a intended to be a basic starting point. Your optimal hyperparams and data may be different.
MODEL_PATH = "models/moondream_base.safetensors"  # TODO: Switch to https://huggingface.co/vikhyatk/moondream2/blob/2025-01-09/model.safetensors
# wget https://huggingface.co/vikhyatk/moondream2/resolve/2025-01-09/model.safetensors . But use starmie or default tokenizer?
LR = 4e-3
EPOCHS = 5
GRAD_ACCUM_STEPS = 1
PLOT_PROGRESS = True
USE_HUGGINGFACE = True


def lr_schedule(step, max_steps, lr):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * lr + 0.9 * lr * x / 0.1
    else:
        return 0.1 * lr + 0.9 * lr * (1 + math.cos(math.pi * (x - 0.1))) / 2


def region_loss(
    hidden_states: torch.Tensor,
    w,
    labels: torch.Tensor,
    c_idx: torch.Tensor,
    s_idx: torch.Tensor,
):
    l_idx = torch.arange(len(labels))

    c_idx = c_idx - 1
    c_hidden = hidden_states[:, c_idx, :]
    c_logits = decode_coordinate(c_hidden, w)
    c_labels = labels[(l_idx % 4) < 2]

    c_loss = F.cross_entropy(
        c_logits.view(-1, c_logits.size(-1)),
        c_labels,
    )

    s_idx = s_idx - 1
    s_hidden = hidden_states[:, s_idx, :]
    s_logits = decode_size(s_hidden, w).view(-1, 1024)
    s_labels = labels[(l_idx % 4) >= 2]

    s_loss = F.cross_entropy(s_logits, s_labels)

    return c_loss + s_loss


class ObjectDetection(Dataset):
    def __init__(
        self, split: str = "train", downsample: bool = False, augment: bool = False
    ):
        self.dataset: datasets.Dataset = datasets.load_dataset(
            "nkasmanoff/retail_detector", split=split
        )
        self.dataset = self.dataset.shuffle(seed=420)

        self.downsample = downsample
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row["image"]
        if self.downsample:
            image = image.resize((image.width // 2, image.height // 2))
        if self.augment:
            # rotate image randomly between 0 and 360 degrees
            if random.random() < 0.5:
                angle = random.randint(0, 360)
                image = image.rotate(angle)
                # adjust box positions
                # each box has coordinates (x, y, width, height) where x and y are the top left corner of the box,
                # given as a percentage of the image width and height
                # now adjust the box positions to rotate with the image
                boxes = row["boxes"]
                for box in boxes:
                    x, y, w, h = box
                    x = x * image.width
                    y = y * image.height
                    x = x * math.cos(angle) - y * math.sin(angle)
                    y = x * math.sin(angle) + y * math.cos(angle)
                    x = x / image.width
                    y = y / image.height
                    box[0] = x
                    box[1] = y
                    box[2] = w
                    box[3] = h
                row["boxes"] = boxes
                # flip image horizontally
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                # adjust box positions
                boxes = row["boxes"]
                for box in boxes:
                    x, y, w, h = box
                    x = 1 - x
                    box[0] = x
                    box[2] = w
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                # adjust box positions
                boxes = row["boxes"]
                for box in boxes:
                    y, x, w, h = box
                    y = 1 - y
                    box[1] = y
                    box[3] = h
                row["boxes"] = boxes

        boxes = row["boxes"]
        labels = row["labels"]

        objects = {}
        for box, label in zip(boxes, labels):
            objects.setdefault(label, []).append(box)

        flat_boxes = []
        class_names = []
        for label, box_list in objects.items():
            for b in box_list:
                flat_boxes.append(b)
                class_names.append(label)

        flat_boxes = torch.as_tensor(flat_boxes, dtype=torch.float16)
        image_id = torch.tensor([idx], dtype=torch.int64)

        return {
            "image": image,
            "boxes": flat_boxes,
            "class_names": class_names,
            "image_id": image_id,
        }


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


def main(args):
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    wandb.init(
        project="retail-detector-ft",
        config={
            "EPOCHS": args.epochs,
            "GRAD_ACCUM_STEPS": args.grad_accum_steps,
            "LR": args.lr,
            "MODEL_PATH": args.model_path,
            "USE_HUGGINGFACE": args.use_huggingface,
            "PLOT_PROGRESS": args.plot_progress,
            "AUGMENT": args.augment,
            "DOWNSAMPLE": args.downsample,
            "FINETUNE_VISION_ENCODER": args.finetune_vision_encoder,
        },
    )
    config = MoondreamConfig()

    if not args.use_huggingface:
        model = MoondreamModel(config)
        load_weights_into_model(args.model_path, model)
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-01-09",
            trust_remote_code=True,  # Uncomment for GPU acceleration & pip install accelerate #
            device_map={"": "cuda"},
        )
        model = hf_model.model

    # If you are struggling with GPU memory, try AdamW8Bit
    params_to_train = [{"params": model.region.parameters()}]
    if args.finetune_vision_encoder:
        params_to_train.append({"params": model.vision_encoder.parameters()})
    optimizer = AdamW(
        params_to_train,
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    train_dataset = ObjectDetection(
        split="train", augment=args.augment, downsample=args.downsample
    )
    val_dataset = ObjectDetection(
        split="train", augment=False, downsample=args.downsample
    )

    total_steps = args.epochs * len(train_dataset) // args.grad_accum_steps
    pbar = tqdm(total=total_steps)

    i = 0
    for epoch in range(args.epochs):
        if args.plot_progress:
            plot_progress_images(epoch, val_dataset, model, args.use_huggingface)

        for sample in train_dataset:
            if len(sample["class_names"]) == 0 or len(sample["boxes"]) == 0:
                continue

            i += 1

            with torch.no_grad() if not args.finetune_vision_encoder else nullcontext():
                img_emb = model._run_vision_encoder(sample["image"])

            with torch.no_grad() if not args.finetune_vision_encoder else nullcontext():
                bos_emb = text_encoder(
                    torch.tensor(
                        [[model.config.tokenizer.bos_id]], device=model.device
                    ),
                    model.text,
                )
                eos_emb = text_encoder(
                    torch.tensor(
                        [[model.config.tokenizer.eos_id]], device=model.device
                    ),
                    model.text,
                )

            boxes_by_class = {}
            for box, cls in zip(sample["boxes"], sample["class_names"]):
                boxes_by_class.setdefault(cls, []).append(box)

            total_loss = 0.0
            for class_name, boxes_list in boxes_by_class.items():
                with (
                    torch.no_grad()
                    if not args.finetune_vision_encoder
                    else nullcontext()
                ):
                    instruction = f"\n\nDetect: {class_name}\n\n"
                    instruction_tokens = model.tokenizer.encode(instruction).ids
                    instruction_emb = text_encoder(
                        torch.tensor([[instruction_tokens]], device=model.device),
                        model.text,
                    ).squeeze(0)

                cs_emb = []
                cs_labels = []
                c_idx = []
                s_idx = []
                for bb in boxes_list:
                    #                    bb = bb.to(dtype=torch.bfloat16, device=model.device)
                    l_cs = len(cs_emb)
                    cs_emb.extend(
                        [
                            encode_coordinate(bb[0].unsqueeze(0), model.region),
                            encode_coordinate(bb[1].unsqueeze(0), model.region),
                            encode_size(bb[2:4], model.region),
                        ]
                    )
                    c_idx.extend([l_cs, l_cs + 1])
                    s_idx.append(l_cs + 2)

                    # Create coordinate bin labels
                    coord_labels = [
                        min(max(torch.round(p * 1023), 0), 1023).item() for p in bb[:2]
                    ]

                    # Create size bin labels using log-scale mapping
                    s_log2_bins = []
                    for s_val in bb[2:4]:
                        s_val = float(s_val)
                        s_clamped = max(s_val, 1 / 1024)
                        s_log2 = math.log2(s_clamped)
                        mapped = (s_log2 + 10.0) / 10.0 * 1023.0
                        s_bin = int(round(mapped))
                        s_bin = max(min(s_bin, 1023), 0)
                        s_log2_bins.append(s_bin)

                    # Combine coordinate and size bin labels
                    cs_labels.extend(coord_labels + s_log2_bins)

                if len(cs_emb) == 0:
                    bb = torch.zeros(4, device=model.device).to(dtype=torch.bfloat16)
                    cs_emb = [
                        encode_coordinate(bb[0].unsqueeze(0), model.region),
                        encode_coordinate(bb[1].unsqueeze(0), model.region),
                        encode_size(bb[2:4], model.region),
                    ]
                    cs_labels = [0, 0, 0, 0, 0, 0]
                    c_idx = [0, 1, 2, 3]
                    s_idx = [4, 5]

                cs_emb = torch.stack(cs_emb)

                inputs_embeds = torch.cat(
                    [bos_emb, img_emb[None], instruction_emb, cs_emb[None], eos_emb],
                    dim=1,
                )
                prefix = inputs_embeds.size(1) - cs_emb.size(0)
                c_idx = torch.tensor(c_idx) + prefix
                s_idx = torch.tensor(s_idx) + prefix

                hidden = _produce_hidden(
                    inputs_embeds=inputs_embeds, w=model.text, config=config.text
                )

                loss = region_loss(
                    hidden_states=hidden,
                    w=model.region,
                    labels=torch.tensor(cs_labels, dtype=torch.int64),
                    c_idx=c_idx,
                    s_idx=s_idx,
                )
                total_loss += loss

            total_loss.backward()

            if i % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr_val = lr_schedule(i / args.grad_accum_steps, total_steps, args.lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_val
                pbar.set_postfix(
                    {"step": i // args.grad_accum_steps, "loss": total_loss.item()}
                )
                pbar.update(1)
                wandb.log(
                    {
                        "loss/train": total_loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
    wandb.finish()

    # Replace with your desired output location.
    save_file(
        model.state_dict(),
        "moondream_finetune_nk.safetensors",
    )

    if args.use_huggingface:
        hf_model.save_pretrained(
            "moondream_finetune_nk_hf"
        )  # save this some other way...
        tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2", revision="2025-01-09", trust_remote_code=True
        )
        tokenizer.save_pretrained("moondream_finetune_nk_hf")


if __name__ == "__main__":
    """
    Replace paths with your appropriate paths.
    To run: python -m moondream.finetune.finetune_region
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/moondream_base.safetensors",
        help="Path to the model safetensors file.",
    )
    parser.add_argument("--lr", type=float, default=4e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--no-plot-progress",
        dest="plot_progress",
        action="store_false",
        help="Do not plot progress images.",
    )
    parser.set_defaults(plot_progress=True)
    parser.add_argument(
        "--no-use-huggingface",
        dest="use_huggingface",
        action="store_false",
        help="Do not use the Hugging Face model.",
    )
    parser.set_defaults(use_huggingface=True)

    parser.add_argument(
        "--finetune-vision-encoder",
        action="store_true",
        help="Enable finetuning of the vision encoder.",
    )

    parser.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Disable data augmentation for the training set.",
    )
    parser.set_defaults(augment=True)
    parser.add_argument(
        "--no-downsample",
        dest="downsample",
        action="store_false",
        help="Do not downsample images.",
    )
    parser.set_defaults(downsample=True)

    args = parser.parse_args()
    main(args)
