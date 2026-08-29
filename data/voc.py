"""Pascal VOC dataset (from scratch; XML parsing, no torchvision refs).

VOC annotations: CC BY 2.5 (data), code here is original.
Returns (image_uint8 HWC RGB, target dict) or transformed tensors in train mode.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import letterbox, clamp_boxes

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def parse_voc_xml(xml_path):
    """Parse one VOC annotation file -> (boxes xyxy np.float32, labels np.int64, difficult np.bool)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = float(size.find("width").text)
    h = float(size.find("height").text)
    boxes, labels, difficult = [], [], []
    for obj in root.iter("object"):
        name = obj.find("name").text.strip().lower()
        if name not in VOC_CLASSES:
            continue
        difficult.append(int(obj.find("difficult").text) if obj.find("difficult") is not None else 0)
        bb = obj.find("bndbox")
        # VOC boxes are 1-indexed
        x1 = max(0.0, float(bb.find("xmin").text) - 1.0)
        y1 = max(0.0, float(bb.find("ymin").text) - 1.0)
        x2 = min(w - 1.0, float(bb.find("xmax").text) - 1.0)
        y2 = min(h - 1.0, float(bb.find("ymax").text) - 1.0)
        boxes.append([x1, y1, x2, y2])
        labels.append(VOC_CLASSES.index(name))
    return (np.array(boxes, dtype=np.float32).reshape(-1, 4),
            np.array(labels, dtype=np.int64),
            np.array(difficult, dtype=np.bool_))


def list_voc_images(root, years=("2007", "2012"), split="trainval"):
    """Return [(img_path, xml_path)] across years."""
    items = []
    for year in years:
        candidates = [os.path.join(root, f"VOC{year}"),
                      os.path.join(root, "VOCdevkit", f"VOC{year}")]
        base = next((c for c in candidates if os.path.isdir(c)), None)
        if base is None:
            raise FileNotFoundError(
                f"VOC{year} not found under {root}; expected one of:\n  " +
                "\n  ".join(candidates))
        txt = os.path.join(base, "ImageSets", "Main", f"{split}.txt")
        with open(txt) as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                img = os.path.join(base, "JPEGImages", f"{name}.jpg")
                xml = os.path.join(base, "Annotations", f"{name}.xml")
                if os.path.exists(img) and os.path.exists(xml):
                    items.append((img, xml))
    return items


class VOCDataset(Dataset):
    """split: trainval | val | test. Test split uses year_test (2007)."""

    def __init__(self, root, years=("2007", "2012"), split="trainval", transform=None):
        self.items = list_voc_images(root, years, split)
        self.transform = transform
        assert len(self.items) > 0, f"no VOC images under {root} for {years}/{split}"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, xml_path = self.items[idx]
        import cv2
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, labels, difficult = parse_voc_xml(xml_path)
        target = {"boxes": torch.from_numpy(boxes), "labels": torch.from_numpy(labels),
                  "difficult": torch.from_numpy(difficult),
                  "orig_size": torch.tensor([img.shape[0], img.shape[1]])}
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    @staticmethod
    def load_image_for_mosaic(img_path):
        import cv2
        img = cv2.imread(img_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


class OverfitSubset(VOCDataset):
    """First N images (with >=1 valid box each) for overfit sanity."""

    def __init__(self, root, n=20, transform=None, years=("2007", "2012")):
        items = list_voc_images(root, years, "trainval")
        picked = []
        for img_path, xml_path in items:
            boxes, labels, _ = parse_voc_xml(xml_path)
            if len(labels) > 0:
                picked.append((img_path, xml_path))
            if len(picked) >= n:
                break
        self.items = picked
        self.transform = transform
        assert len(self.items) == n, f"only found {len(self.items)}/{n} images with boxes"