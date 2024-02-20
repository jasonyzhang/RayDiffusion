import os.path as osp
from glob import glob

import torch
from omegaconf import OmegaConf

from ray_diffusion.model.diffuser import RayDiffuser
from ray_diffusion.model.scheduler import NoiseScheduler


def load_model(
    output_dir, checkpoint=None, device="cuda:0", custom_keys=None, ignore_keys=()
):
    """
    Loads a model and config from an output directory.

    E.g. to load with different number of images,
    ```
    custom_keys={"model.num_images": 15}, ignore_keys=["pos_table"]
    ```

    Args:
        output_dir (str): Path to the output directory.
        checkpoint (str or int): Path to the checkpoint to load. If None, loads the
            latest checkpoint.
        device (str): Device to load the model on.
        custom_keys (dict): Dictionary of custom keys to override in the config.
    """
    if checkpoint is None:
        checkpoint_path = sorted(glob(osp.join(output_dir, "checkpoints", "*.pth")))[-1]
    else:
        if isinstance(checkpoint, int):
            checkpoint_name = f"ckpt_{checkpoint:08d}.pth"
        else:
            checkpoint_name = checkpoint
        checkpoint_path = osp.join(output_dir, "checkpoints", checkpoint_name)
    print("Loading checkpoint", osp.basename(checkpoint_path))

    cfg = OmegaConf.load(osp.join(output_dir, "hydra", "config.yaml"))
    if custom_keys is not None:
        for k, v in custom_keys.items():
            OmegaConf.update(cfg, k, v)
    noise_scheduler = NoiseScheduler(
        type=cfg.noise_scheduler.type,
        max_timesteps=cfg.noise_scheduler.max_timesteps,
        beta_start=cfg.noise_scheduler.beta_start,
        beta_end=cfg.noise_scheduler.beta_end,
    )

    model = RayDiffuser(
        depth=cfg.model.depth,
        width=cfg.model.num_patches_x,
        P=1,
        max_num_images=cfg.model.num_images,
        noise_scheduler=noise_scheduler,
        feature_extractor=cfg.model.feature_extractor,
        append_ndc=cfg.model.append_ndc,
    ).to(device)

    data = torch.load(checkpoint_path)
    state_dict = {}
    for k, v in data["state_dict"].items():
        include = True
        for ignore_key in ignore_keys:
            if ignore_key in k:
                include = False
        if include:
            state_dict[k] = v

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        print("Missing keys:", missing)
    if len(unexpected) > 0:
        print("Unexpected keys:", unexpected)
    model = model.eval()
    return model, cfg
