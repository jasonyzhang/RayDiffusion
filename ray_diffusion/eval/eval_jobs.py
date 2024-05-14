"""
python -m ray_diffusion.eval.eval_jobs --eval_type diffusion --eval_path models/co3d_diffusion
"""

import argparse
import itertools
import json
import os
from glob import glob

import numpy as np
from tqdm.auto import tqdm

from ray_diffusion.dataset.co3d_v2 import TEST_CATEGORIES, TRAINING_CATEGORIES
from ray_diffusion.eval.eval_category import save_results


def evaluate_ray_diffusion(eval_path):
    JOB_PARAMS = {
        "output_dir": [eval_path],
        "checkpoint": [450_000],
        "num_images": [2, 3, 4, 5, 6, 7, 8],
        "category": TRAINING_CATEGORIES + TEST_CATEGORIES,
        "calculate_additional_timesteps": [True],
        "sample_num": [0, 1, 2, 3, 4],
        "rescale_noise": ["zero"],  # Don't add noise during DDPM
        "normalize_moments": [True],
    }
    keys, values = zip(*JOB_PARAMS.items())
    job_configs = [dict(zip(keys, p)) for p in itertools.product(*values)]
    for job_config in tqdm(job_configs):
        # You may want to parallelize these jobs here, e.g. with submitit.
        save_results(**job_config)


def evaluate_ray_regression(eval_path):
    JOB_PARAMS = {
        "output_dir": [eval_path],
        "checkpoint": [300_000],
        "num_images": [2, 3, 4, 5, 6, 7, 8],
        "category": TRAINING_CATEGORIES + TEST_CATEGORIES,
        "sample_num": [0, 1, 2, 3, 4],
    }
    keys, values = zip(*JOB_PARAMS.items())
    job_configs = [dict(zip(keys, p)) for p in itertools.product(*values)]
    for job_config in tqdm(job_configs[::-1]):
        # You may want to parallelize these jobs here, e.g. with submitit.
        save_results(**job_config)


def process_predictions(eval_path, pred_index):
    """
    pred_index should be 0 for regression and 7 for diffusion (corresponding to T=30)
    """
    errors = {
        c: {n: {"CC": [], "R": []} for n in range(2, 9)}
        for c in TRAINING_CATEGORIES + TEST_CATEGORIES
    }

    for category in tqdm(TRAINING_CATEGORIES + TEST_CATEGORIES):
        for num_images in range(2, 9):
            for sample_num in range(5):
                data_path = glob(
                    os.path.join(
                        eval_path,
                        "eval",
                        f"{category}_{num_images}_{sample_num}_ckpt*.json",
                    )
                )[0]
                with open(data_path) as f:
                    eval_data = json.load(f)

                for preds in eval_data.values():
                    errors[category][num_images]["R"].extend(
                        preds[pred_index]["R_error"]
                    )
                    errors[category][num_images]["CC"].extend(
                        preds[pred_index]["CC_error"]
                    )

    threshold_R = 15
    threshold_CC = 0.1

    all_seen_acc_R = []
    all_seen_acc_CC = []
    all_unseen_acc_R = []
    all_unseen_acc_CC = []

    for num_images in range(2, 9):
        seen_acc_R = []
        seen_acc_CC = []
        unseen_acc_R = []
        unseen_acc_CC = []
        for category in TEST_CATEGORIES:
            unseen_acc_R.append(
                np.mean(np.array(errors[category][num_images]["R"]) < threshold_R)
            )
            unseen_acc_CC.append(
                np.mean(np.array(errors[category][num_images]["CC"]) < threshold_CC)
            )

        for category in TRAINING_CATEGORIES:
            seen_acc_R.append(
                np.mean(np.array(errors[category][num_images]["R"]) < threshold_R)
            )
            seen_acc_CC.append(
                np.mean(np.array(errors[category][num_images]["CC"]) < threshold_CC)
            )

        all_seen_acc_R.append(np.mean(seen_acc_R))
        all_seen_acc_CC.append(np.mean(seen_acc_CC))
        all_unseen_acc_R.append(np.mean(unseen_acc_R))
        all_unseen_acc_CC.append(np.mean(unseen_acc_CC))

    print("N=       ", " ".join(f"{i: 5}" for i in range(2, 9)))
    print("Seen R   ", " ".join([f"{x:0.3f}" for x in all_seen_acc_R]))
    print("Seen CC  ", " ".join([f"{x:0.3f}" for x in all_seen_acc_CC]))
    print("Unseen R ", " ".join([f"{x:0.3f}" for x in all_unseen_acc_R]))
    print("Unseen CC", " ".join([f"{x:0.3f}" for x in all_unseen_acc_CC]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_type", type=str, default="diffusion", help="diffusion or regression"
    )
    parser.add_argument("--eval_path", type=str, default=None)
    args = parser.parse_args()

    eval_type = args.eval_type
    eval_path = args.eval_path
    if eval_path is None:
        if eval_type == "diffusion":
            eval_path = "models/co3d_diffusion"
        elif eval_type == "regression":
            eval_path = "models/co3d_regression"

    if eval_type == "diffusion":
        evaluate_ray_diffusion(eval_path)
        process_predictions(eval_path, 7)
    elif eval_type == "regression":
        evaluate_ray_regression(eval_path)
        process_predictions(eval_path, 0)
    else:
        raise Exception(f"Unknown eval_type: {eval_type}")
