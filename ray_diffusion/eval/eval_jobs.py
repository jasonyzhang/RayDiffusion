from ray_diffusion.dataset.co3d_v2 import TEST_CATEGORIES, TRAINING_CATEGORIES


def evaluate_ray_diffusion():
    JOB_PARAMS = {
        "output_dir": ["models/co3d_diffusion"],
        "checkpoint": [450_000],
        "num_images": [2, 3, 4, 5, 6, 7, 8],
        "category": TRAINING_CATEGORIES + TEST_CATEGORIES,
        "calculate_additional_timesteps": [True],
        "sample_num": [0, 1, 2, 3, 4],
        "rescale_noise": ["zero"],  # Don't add noise during DDPM
        "normalize_moments": [True],
    }


def evaluate_ray_regression():
    JOB_PARAMS = {}
