import json
import os

import numpy as np
import torch
from tqdm.auto import tqdm

from ray_diffusion.dataset.co3d_v2 import Co3dDataset
from ray_diffusion.eval.utils import (
    compute_angular_error_batch,
    compute_camera_center_error,
    full_scene_scale,
    n_to_np_rotations,
)
from ray_diffusion.inference.load_model import load_model
from ray_diffusion.inference.predict import predict_cameras


@torch.no_grad()
def evaluate(
    cfg,
    model,
    dataset,
    num_images,
    device,
    use_pbar=True,
    calculate_intrinsics=False,
    additional_timesteps=(),
    use_beta_tilde=False,
    normalize_moments=True,
    rescale_noise="zero",
    max_num_images=None,
):
    results = {}
    instances = np.arange(0, len(dataset))

    instances = tqdm(instances) if use_pbar else instances

    for counter, idx in enumerate(instances):
        batch = dataset[idx]
        instance = batch["model_id"]
        images = batch["image"].to(device)
        focal_length = batch["focal_length"].to(device)[:num_images]
        R = batch["R"].to(device)[:num_images]
        T = batch["T"].to(device)[:num_images]
        crop_parameters = batch["crop_parameters"].to(device)[:num_images]

        pred_cameras, additional_cams = predict_cameras(
            model,
            images,
            device,
            pred_x0=cfg.model.pred_x0,
            crop_parameters=crop_parameters,
            num_patches_x=cfg.model.num_patches_x,
            num_patches_y=cfg.model.num_patches_y,
            additional_timesteps=additional_timesteps,
            calculate_intrinsics=calculate_intrinsics,
            use_beta_tilde=use_beta_tilde,
            normalize_moments=normalize_moments,
            rescale_noise=rescale_noise,
            use_regression=cfg.training.regression,
            max_num_images=max_num_images,
        )

        cameras_to_evaluate = additional_cams + [pred_cameras]

        all_cams_batch = dataset.get_data(
            sequence_name=instance, ids=np.arange(0, batch["n"]), no_images=True
        )
        gt_scene_scale = full_scene_scale(all_cams_batch)
        R_gt = R
        T_gt = T

        errors = []
        for camera in cameras_to_evaluate:
            R_pred = camera.R
            T_pred = camera.T
            f_pred = camera.focal_length

            R_pred_rel = n_to_np_rotations(num_images, R_pred).cpu().numpy()
            R_gt_rel = n_to_np_rotations(num_images, batch["R"]).cpu().numpy()
            R_error = compute_angular_error_batch(R_pred_rel, R_gt_rel)

            CC_error, _ = compute_camera_center_error(
                R_pred, T_pred, R_gt, T_gt, gt_scene_scale
            )

            errors.append(
                {
                    "R_pred": R_pred.detach().cpu().numpy().tolist(),
                    "T_pred": T_pred.detach().cpu().numpy().tolist(),
                    "f_pred": f_pred.detach().cpu().numpy().tolist(),
                    "R_gt": R_gt.detach().cpu().numpy().tolist(),
                    "T_gt": T_gt.detach().cpu().numpy().tolist(),
                    "f_gt": focal_length.detach().cpu().numpy().tolist(),
                    "scene_scale": gt_scene_scale,
                    "R_error": R_error.tolist(),
                    "CC_error": CC_error,
                }
            )
        results[instance] = errors
        if counter == len(dataset) - 1:
            break
    return results


def save_results(
    output_dir,
    checkpoint=450000,
    category="hydrant",
    num_images=None,
    calculate_additional_timesteps=False,
    calculate_intrinsics=False,
    split="test",
    force=False,
    sample_num=1,
    use_beta_tilde=False,
    normalize_moments=False,
    rescale_noise="square_root",
    max_num_images=None,
):
    eval_path = os.path.join(
        output_dir,
        "eval",
        f"{category}_{num_images}_{sample_num}_ckpt{checkpoint}.json",
    )

    if os.path.exists(eval_path) and not force:
        print(f"File {eval_path} already exists. Skipping.")
        return

    if num_images > 8:
        custom_keys = {"model.num_images": num_images}
        ignore_keys = ["pos_table"]
    else:
        custom_keys = None
        ignore_keys = []

    device = torch.device("cuda")
    model, cfg = load_model(
        output_dir,
        checkpoint=checkpoint,
        device=device,
        custom_keys=custom_keys,
        ignore_keys=ignore_keys,
    )
    if num_images is None:
        num_images = cfg.model.num_images

    dataset = Co3dDataset(
        category=category,
        split=split,
        num_images=num_images,
        apply_augmentation=True,
        sample_num=None if split == "train" else sample_num,
    )
    print(f"Category {category} {len(dataset)}")

    if calculate_additional_timesteps:
        additional_timesteps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    else:
        additional_timesteps = []

    results = evaluate(
        cfg=cfg,
        model=model,
        dataset=dataset,
        num_images=num_images,
        device=device,
        calculate_intrinsics=calculate_intrinsics,
        additional_timesteps=additional_timesteps,
        use_beta_tilde=use_beta_tilde,
        normalize_moments=normalize_moments,
        rescale_noise=rescale_noise,
        max_num_images=max_num_images,
    )

    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(results, f)
