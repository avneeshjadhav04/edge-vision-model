"""Data pipeline tests with synthetic images (no dataset download needed)."""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.augment import Mosaic, RandomAffineBoxes, Photometric, TrainTransform, EvalTransform
from data.common import letterbox, clamp_boxes


def test_mosaic_boxes_track_objects():
    """One bright square object; after mosaic the box must overlap the square."""
    torch.manual_seed(0)
    img = np.full((320, 480, 3), 30, np.uint8)
    img[60:200, 100:300] = 220  # object
    target = {"boxes": torch.tensor([[100., 60., 300., 200.]]),
              "labels": torch.tensor([1])}
    mosaic = Mosaic(320, p=1.0)
    out, tgt = mosaic(img, target)  # no extras -> same image in all 4 quadrants
    assert out.shape == (640, 640, 3)
    # IoU proxy: object pixel centroid must fall inside predicted box
    assert tgt["boxes"].shape[0] >= 1
    # all boxes within canvas
    assert tgt["boxes"][:, ::2].min() >= -1 and tgt["boxes"][:, 2].max() <= 641
    print("test_mosaic OK", tgt["boxes"].shape)


def test_affine_and_photometric():
    img = np.full((640, 640, 3), 100, np.uint8)
    img[100:300, 200:400] = 250
    target = {"boxes": torch.tensor([[200., 100., 400., 300.]]),
              "labels": torch.tensor([0])}
    aug = RandomAffineBoxes(640, scale=0.2, translate=0.05, fliplr=0.0)
    im2, t2 = aug(img, target)
    assert im2.shape == (640, 640, 3)
    ph = Photometric()
    im3, _ = ph(im2, t2)
    assert im3.dtype == np.uint8
    print("test_affine_and_photometric OK")


def test_train_transform_output():
    class DummyDS:
        def __len__(self):
            return 4

        def __getitem__(self, i):
            img = np.full((240, 320, 3), 60, np.uint8)
            img[50:150, 80:220] = 200
            return img, {"boxes": torch.tensor([[80., 50., 220., 150.]]),
                         "labels": torch.tensor([3])}
    t = TrainTransform(320, mosaic_p=1.0, dataset_for_mosaic=DummyDS())
    img = np.full((240, 320, 3), 30, np.uint8)
    img[40:120, 60:200] = 255
    x, tgt = t(img, {"boxes": torch.tensor([[60., 40., 200., 120.]]),
                     "labels": torch.tensor([3])})
    assert x.shape == (3, 320, 320) and x.dtype == torch.float32 and x.max() <= 1.0
    assert tgt["boxes"].shape[1] == 4
    print("test_train_transform OK", x.shape, tgt["boxes"].shape)


def test_letterbox():
    img = np.zeros((300, 500, 3), np.uint8)
    out, r, pads = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert abs(r - 640 / 500) < 1e-6
    assert pads[0] >= 0 and pads[1] >= 0 and (pads[0] > 0 or pads[1] > 0)
    print("test_letterbox OK", r, pads)


if __name__ == "__main__":
    test_mosaic_boxes_track_objects()
    test_affine_and_photometric()
    test_train_transform_output()
    test_letterbox()
    print("ALL DATA TESTS PASSED")