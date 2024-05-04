"""
python -m ray_diffusion.eval.eval_jobs --eval_type diffusion --eval_path models/co3d_diffusion
"""

import argparse
import itertools

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
    for job_config in tqdm(job_configs):
        # You may want to parallelize these jobs here, e.g. with submitit.
        save_results(**job_config)


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
    elif eval_type == "regression":
        evaluate_ray_regression(eval_path)
    else:
        raise Exception(f"Unknown eval_type: {eval_type}")
