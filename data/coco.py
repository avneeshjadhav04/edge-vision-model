"""COCO dataset (from scratch; direct JSON handling, pycocotools only for eval).

COCO annotations: CC BY 4.0 (data). Targets use xyxy pixel boxes; categories are
mapped to contiguous ids 0..79 (contiguous_cat_ids provided by data/coco_io.py).
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import clamp_boxes


class CocoDataset(Dataset):
    def __init__(self, root, split="train2017", year=2017, transform=None,
                 min_area=1.0, load_on_demand=True):
        from .coco_io import load_coco_annotations
        self.root = root
        self.split = split
        self.transform = transform
        ann_file = os.path.join(root, "annotations", f"instances_{split}.json")
        self.index = load_coco_annotations(ann_file)  # builds id -> (file_name, boxes, labels)
        self.ids = list(self.index.keys())
        self.img_dir = os.path.join(root, split)
        self.min_area = min_area

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        import cv2
        img_id = self.ids[idx]
        rec = self.index[img_id]
        path = os.path.join(self.img_dir, rec["file_name"])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = rec["boxes"].clone()
        labels = rec["labels"].clone()
        h, w = img.shape[:2]
        keep = clamp_boxes(boxes, w, h)
        target = {"boxes": boxes[keep], "labels": labels[keep],
                  "image_id": torch.tensor(img_id),
                  "orig_size": torch.tensor([h, w])}
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target