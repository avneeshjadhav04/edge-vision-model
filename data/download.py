"""Dataset downloaders (VOC: host.tar scripts; COCO: official zip URLs)."""
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile

VOC_URLS = {
    "2007": {
        "trainval": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "test": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
    },
    "2012": {
        "trainval": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
    },
}

COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "ann2017": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0 and downloaded % (10 * 1024 * 1024) < block_size:
        print(f"\r  {downloaded / 1e6:.0f}/{total_size / 1e6:.0f} MB", end="", flush=True)


def download(url, dest_dir, filename=None):
    os.makedirs(dest_dir, exist_ok=True)
    fn = filename or url.split("/")[-1]
    path = os.path.join(dest_dir, fn)
    if not os.path.exists(path):
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path, reporthook=_progress)
        print()
    return path


def download_voc(root="./datasets/VOC", years=("2007", "2012"), with_test=True):
    for y in years:
        tars = [("trainval", VOC_URLS[y]["trainval"])]
        if with_test and y == "2007":
            tars.append(("test", VOC_URLS["2007"]["test"]))
        for split, url in tars:
            path = download(url, root)
            print(f"extracting {path}")
            with tarfile.open(path) as t:
                t.extractall(root)
    print("VOC ready at", os.path.abspath(root))


def download_coco(root="./datasets/coco", splits=("val2017",), with_ann=True,
                  with_train=False):
    os.makedirs(root, exist_ok=True)
    if with_train and "train2017" not in splits:
        splits = tuple(splits) + ("train2017",)
    for s in splits:
        path = download(COCO_URLS[s], root)
        print(f"extracting {path}")
        with zipfile.ZipFile(path) as z:
            z.extractall(root)
    if with_ann and not os.path.exists(os.path.join(root, "annotations")):
        path = download(COCO_URLS["ann2017"], root)
        print(f"extracting {path}")
        with zipfile.ZipFile(path) as z:
            z.extractall(root)
    print("COCO ready at", os.path.abspath(root))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "voc":
        download_voc()
    elif args[0] == "coco-val":
        download_coco(splits=("val2017",))
    elif args[0] == "coco-full":
        download_coco(splits=("val2017",), with_train=True)
    else:
        print("usage: python -m data.download [voc|coco-val|coco-full]")