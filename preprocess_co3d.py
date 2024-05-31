"""
Script to pre-process camera poses and bounding boxes for CO3Dv2 dataset. This is
important because computing the bounding boxes from the masks is a significant
bottleneck.

First, you should pre-compute the bounding boxes since this takes a long time.

Usage:
    python -m preprocess.preprocess_co3d --category all --precompute_bbox \
        --co3d_v2_dir /path/to/co3d_v2
    python -m preprocess.preprocess_co3d --category all \
        --co3d_v2_dir /path/to/co3d_v2
"""

import argparse
import gzip
import json
import os.path as osp
from glob import glob

import ipdb
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# fmt: off
CATEGORIES = [
    "apple", "backpack", "ball", "banana", "baseballbat", "baseballglove",
    "bench", "bicycle", "book", "bottle", "bowl", "broccoli", "cake", "car", "carrot",
    "cellphone", "chair", "couch", "cup", "donut", "frisbee", "hairdryer", "handbag",
    "hotdog", "hydrant", "keyboard", "kite", "laptop", "microwave", "motorcycle",
    "mouse", "orange", "parkingmeter", "pizza", "plant", "remote", "sandwich",
    "skateboard", "stopsign", "suitcase", "teddybear", "toaster", "toilet", "toybus",
    "toyplane", "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass",
]
# fmt: on


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="apple")
    parser.add_argument("--output_dir", type=str, default="data/co3d_v2_annotations")
    parser.add_argument("--co3d_v2_dir", type=str, default="data/co3d_v2")
    parser.add_argument(
        "--min_quality",
        type=float,
        default=0.5,
        help="Minimum viewpoint quality score.",
    )
    parser.add_argument("--precompute_bbox", action="store_true")
    return parser


def mask_to_bbox(mask):
    """
    xyxy format
    """
    mask = mask > 0.4
    if not np.any(mask):
        return []
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax) + 1, int(rmax) + 1]


def precompute_bbox(co3d_dir, category, output_dir):
    """
    Precomputes bounding boxes for all frames using the masks. This can be an expensive
    operation because it needs to load every mask in the dataset. Thus, we only want to
    run this once, whereas processing the rest of the dataset is fast.
    """
    category_dir = osp.join(co3d_dir, category)
    print("Precomputing bbox for:", category)
    all_masks = sorted(glob(osp.join(category_dir, "*", "masks", "*.png")))
    bboxes = {}
    for mask_filename in tqdm(all_masks):
        mask = plt.imread(mask_filename)
        # /Dataset/category/sequence/masks/mask.png -> category/sequence/mask/mask.png
        mask_filename = mask_filename.replace(osp.dirname(category_dir), "")[1:]
        try:
            bboxes[mask_filename] = mask_to_bbox(mask)
        except IndexError:
            ipdb.set_trace()
    output_file = osp.join(output_dir, f"{category}_bbox.jgz")
    with gzip.open(output_file, "w") as f:
        f.write(json.dumps(bboxes).encode("utf-8"))


def process_poses(co3d_dir, category, output_dir, min_quality):
    category_dir = osp.join(co3d_dir, args.category)
    print("Processing category:", category)
    frame_file = osp.join(category_dir, "frame_annotations.jgz")
    sequence_file = osp.join(category_dir, "sequence_annotations.jgz")
    subset_lists_file = osp.join(category_dir, "set_lists/set_lists_fewview_dev.json")

    bbox_file = osp.join(output_dir, f"{category}_bbox.jgz")

    with open(subset_lists_file) as f:
        subset_lists_data = json.load(f)

    with gzip.open(sequence_file, "r") as fin:
        sequence_data = json.loads(fin.read())

    with gzip.open(frame_file, "r") as fin:
        frame_data = json.loads(fin.read())

    with gzip.open(bbox_file, "r") as fin:
        bbox_data = json.loads(fin.read())

    frame_data_processed = {}
    for f_data in frame_data:
        sequence_name = f_data["sequence_name"]
        if sequence_name not in frame_data_processed:
            frame_data_processed[sequence_name] = {}
        frame_data_processed[sequence_name][f_data["frame_number"]] = f_data

    good_quality_sequences = set()
    for seq_data in sequence_data:
        if seq_data["viewpoint_quality_score"] > min_quality:
            good_quality_sequences.add(seq_data["sequence_name"])

    for subset in ["train", "test"]:
        category_data = {}  # {sequence_name: [{filepath, R, T}]}
        for seq_name, frame_number, filepath in subset_lists_data[subset]:
            if seq_name not in good_quality_sequences:
                continue

            if seq_name not in category_data:
                category_data[seq_name] = []

            mask_path = filepath.replace("images", "masks").replace(".jpg", ".png")
            bbox = bbox_data[mask_path]
            if bbox == []:
                # Mask did not include any object.
                continue

            frame_data = frame_data_processed[seq_name][frame_number]
            category_data[seq_name].append(
                {
                    "filepath": filepath,
                    "R": frame_data["viewpoint"]["R"],
                    "T": frame_data["viewpoint"]["T"],
                    "focal_length": frame_data["viewpoint"]["focal_length"],
                    "principal_point": frame_data["viewpoint"]["principal_point"],
                    "bbox": bbox,
                }
            )

        output_file = osp.join(args.output_dir, f"{args.category}_{subset}.jgz")
        with gzip.open(output_file, "w") as f:
            f.write(json.dumps(category_data).encode("utf-8"))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.category == "all":
        categories = CATEGORIES
    else:
        categories = [args.category]
    if args.precompute_bbox:
        for category in categories:
            precompute_bbox(
                co3d_dir=args.co3d_v2_dir,
                category=category,
                output_dir=args.output_dir,
            )
    else:
        for category in categories:
            process_poses(
                co3d_dir=args.co3d_v2_dir,
                category=category,
                output_dir=args.output_dir,
                min_quality=args.min_quality,
            )
