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

from ..torch.weights import load_weights_into_model
from ..torch.moondream import MoondreamModel, MoondreamConfig, text_encoder
from ..torch.text import _produce_hidden
from ..torch.region import (
    decode_coordinate,
    decode_size,
    encode_coordinate,
    encode_size,
)
from .region_finetune_utils import plot_progress_images


# This is a intended to be a basic starting point. Your optimal hyperparams and data may be different.
RANDOM_SEED = 250
MODEL_PATH = ""
#LR = 3e-4  # Learning rate
LR = 9e-5

EPOCHS = 50
GRAD_ACCUM_STEPS = 1
PLOT_PROGRESS = True
USE_HUGGINGFACE = True
#REVISION = "2025-01-09"  # "2025-04-14"
REVISION="2025-06-21"
DATASET_NAME = (
    "nkasmanoff/retail_detector_flattened"  # "nkasmanoff/retail_detector_flattened"
)


def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2


def ciou_loss(boxes1, boxes2, eps=1e-7):
    """
    Calculate Complete Intersection over Union (CIoU) loss.
    boxes1, boxes2: [N, 4] tensors with format (x_center, y_center, width, height)
    """
    # Convert boxes to (x1, y1, x2, y2)
    b1_x1, b1_y1 = boxes1[:, 0] - boxes1[:, 2] / 2, boxes1[:, 1] - boxes1[:, 3] / 2
    b1_x2, b1_y2 = boxes1[:, 0] + boxes1[:, 2] / 2, boxes1[:, 1] + boxes1[:, 3] / 2
    b2_x1, b2_y1 = boxes2[:, 0] - boxes2[:, 2] / 2, boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2, b2_y2 = boxes2[:, 0] + boxes2[:, 2] / 2, boxes2[:, 1] + boxes2[:, 3] / 2

    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1, min=0
    )

    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + eps

    # IoU
    iou = inter_area / union_area

    # Distance penalty
    center_dist = torch.sum((boxes1[:, :2] - boxes2[:, :2]) ** 2, dim=-1)
    enclosing_x1 = torch.min(b1_x1, b2_x1)
    enclosing_y1 = torch.min(b1_y1, b2_y1)
    enclosing_x2 = torch.max(b1_x2, b2_x2)
    enclosing_y2 = torch.max(b1_y2, b2_y2)
    enclosing_diag = (enclosing_x2 - enclosing_x1) ** 2 + (
        enclosing_y2 - enclosing_y1
    ) ** 2
    distance_penalty = center_dist / (enclosing_diag + eps)

    # Aspect ratio penalty
    v = (4 / math.pi**2) * torch.pow(
        torch.atan(boxes1[:, 2] / (boxes1[:, 3] + eps))
        - torch.atan(boxes2[:, 2] / (boxes2[:, 3] + eps)),
        2,
    )
    with torch.no_grad():
        alpha = v / ((1 - iou) + v + eps)
    aspect_ratio_penalty = alpha * v

    return torch.mean(1 - iou + distance_penalty + aspect_ratio_penalty)


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
    print("c_labels", c_labels)

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
        self, split: str = "train", downsample: bool = False, overfit_batch: bool = True
    ):
        self.dataset: datasets.Dataset = datasets.load_dataset(
            DATASET_NAME, split=split
        )
        self.dataset = self.dataset.shuffle(seed=RANDOM_SEED)
        if overfit_batch:
            # take a single row from dataset
            self.dataset = self.dataset.select([0])
        self.downsample = downsample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row["image"]
        if self.downsample:
            image = image.resize((image.width // 2, image.height // 2))

        boxes = row["boxes"][:1]
        labels = row["labels"][:1]

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


def main():
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    wandb.init(
        project="retail-detector-ft",
        config={
            "EPOCHS": EPOCHS,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "LR": LR,
            "MODEL_PATH": MODEL_PATH,
            "USE_HUGGINGFACE": USE_HUGGINGFACE,
            "PLOT_PROGRESS": PLOT_PROGRESS,
        },
    )
    config = MoondreamConfig()

    if not USE_HUGGINGFACE:
        model = MoondreamModel(config)
        load_weights_into_model(MODEL_PATH, model)
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision=REVISION,  # later revisions fail with  AttributeError: 'ModuleDict' object has no attribute 'kv_cache'
            trust_remote_code=True,  # Uncomment for GPU acceleration & pip install accelerate #
            device_map={"": "cuda"},
        )
        model = hf_model.model

    # If you are struggling with GPU memory, try AdamW8Bit
    optimizer = AdamW(
        [{"params": model.parameters()}],
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    train_dataset = ObjectDetection(split="train")
    val_dataset = ObjectDetection(split="train")

    total_steps = EPOCHS * len(train_dataset) // GRAD_ACCUM_STEPS
    pbar = tqdm(total=total_steps)

    i = 0
    for epoch in range(EPOCHS):
        if PLOT_PROGRESS:
            model.eval();
            plot_progress_images(epoch, val_dataset, hf_model, USE_HUGGINGFACE)
        model.train();
        for sample in train_dataset:
            if len(sample["class_names"]) == 0 or len(sample["boxes"]) == 0:
                continue
            i += 1
            with torch.no_grad():
                img_emb = model._run_vision_encoder(sample["image"])
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
                with torch.no_grad():
                    print(f"Processing class: {class_name}")
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
                    bb = bb.to(dtype=torch.bfloat16, device=model.device)
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
                    continue
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

            if total_loss == 0.0:
                continue

            total_loss.backward()

            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr_val = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_val
                pbar.set_postfix(
                    {"step": i // GRAD_ACCUM_STEPS, "loss": total_loss.item()}
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
    if not USE_HUGGINGFACE:
        save_file(
            model.state_dict(),
            "moondream_finetune_nk.safetensors",
        )

    if USE_HUGGINGFACE:
        hf_model.push_to_hub(
            "nkasmanoff/moondream_finetune_nk_hf"
        )  # save this some other way...
        tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2", revision=REVISION, trust_remote_code=True
        )
        tokenizer.push_to_hub("nkasmanoff/moondream_finetune_nk_hf")


if __name__ == "__main__":
    """
    Replace paths with your appropriate paths.
    To run: python -m moondream.finetune.finetune_region
    """
    main()
