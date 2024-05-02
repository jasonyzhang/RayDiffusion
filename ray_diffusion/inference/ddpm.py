import ipdb  # noqa: F401
import torch
from tqdm.auto import tqdm

from ray_diffusion.utils.rays import compute_ndc_coordinates

rescale_fn = {
    "zero": lambda x: 0,
    "identity": lambda x: x,
    "square": lambda x: x**2,
    "square_root": lambda x: torch.sqrt(x),
}


def inference_ddpm(
    model,
    images,
    device,
    crop_parameters=None,
    pbar=True,
    visualize=False,
    clip_bounds_m=5,
    clip_bounds_d=5,
    pred_x0=False,
    stop_iteration=-1,
    use_beta_tilde=False,
    num_patches_x=16,
    num_patches_y=16,
    normalize_directions=True,
    normalize_moments=False,
    rescale_noise="square_root",
    compute_x0=False,
    max_num_images=None,
):
    """
    Implements DDPM-style inference.

    To get multiple samples, batch the images multiple times.

    Args:
        model: Ray Diffuser.
        images (torch.Tensor): (B, N, C, H, W).
        crop_parameters (torch.Tensor): (B, N, 4) or None.
        pbar (bool): If True, shows a progress bar.
    """

    num_train_steps = model.noise_scheduler.max_timesteps
    batch_size = images.shape[0]
    num_images = images.shape[1]
    images = images.to(device)

    with torch.no_grad():
        x_t = torch.randn(
            batch_size, num_images, 6, num_patches_x, num_patches_y, device=device
        )
        if visualize:
            x_ts = [x_t]
            all_pred = []
            noise_samples = []
        image_features = model.feature_extractor(images, autoresize=True)
        if model.append_ndc:
            # (B, N, H, W, 3)
            ndc_coordinates = compute_ndc_coordinates(
                crop_parameters=crop_parameters,
                num_patches_x=num_patches_x,
                num_patches_y=num_patches_y,
            )
            ndc_coordinates = ndc_coordinates.to(device)[..., :2]
            # (B, N, 2, H, W)
            ndc_coordinates = ndc_coordinates.permute(0, 1, 4, 2, 3)
        else:
            ndc_coordinates = None

        loop = range(num_train_steps - 1, stop_iteration, -1)
        if pbar:
            loop = tqdm(loop)
        for t in loop:
            z = (
                torch.randn(
                    batch_size,
                    num_images,
                    6,
                    num_patches_x,
                    num_patches_y,
                    device=device,
                )
                if t > 0
                else 0
            )
            if max_num_images is None:
                eps_pred, noise_sample = model(
                    features=image_features,
                    rays_noisy=x_t,
                    t=t,
                    ndc_coordinates=ndc_coordinates,
                    compute_x0=compute_x0,
                )
            else:
                eps_pred = torch.zeros_like(x_t)
                noise_sample = torch.zeros_like(x_t)
                indices_split = torch.split(
                    torch.randperm(num_images - 1) + 1, max_num_images - 1
                )
                for indices in indices_split:
                    indices = torch.cat((torch.tensor([0]), indices))
                    eps_pred_ind, noise_sample_ind = model(
                        features=image_features[:, indices],
                        rays_noisy=x_t[:, indices],
                        t=t,
                        ndc_coordinates=ndc_coordinates[:, indices],
                        compute_x0=compute_x0,
                    )
                    eps_pred[:, indices] += eps_pred_ind
                    if noise_sample_ind is not None:
                        noise_sample[:, indices] += noise_sample_ind
                eps_pred[:, 0] /= len(indices_split)  # Take average for first one
                noise_sample[:, 0] /= len(indices_split)

            if pred_x0:
                c = torch.linalg.norm(eps_pred[:, :, :3], dim=2, keepdim=True)
                d = eps_pred[:, :, :3]
                m = eps_pred[:, :, 3:]
                if normalize_directions:
                    eps_pred[:, :, :3] = d / c
                if normalize_moments:
                    eps_pred[:, :, 3:] = m - (d * m).sum(dim=2, keepdim=True) * d / c

            if visualize:
                all_pred.append(eps_pred.clone())
                noise_samples.append(noise_sample)

            alpha_t = model.noise_scheduler.alphas[t]
            alpha_bar_t = model.noise_scheduler.alphas_cumprod[t]
            if t == 0:
                alpha_bar_tm1 = torch.ones_like(alpha_bar_t)
            else:
                alpha_bar_tm1 = model.noise_scheduler.alphas_cumprod[t - 1]
            beta_t = model.noise_scheduler.betas[t]
            if use_beta_tilde:
                sigma_t = (1 - alpha_bar_tm1) / (1 - alpha_bar_t) * beta_t
            else:
                sigma_t = beta_t

            sigma_t = rescale_fn[rescale_noise](sigma_t)

            if pred_x0:
                w_x0 = torch.sqrt(alpha_bar_tm1) * beta_t / (1 - alpha_bar_t)
                w_xt = torch.sqrt(alpha_t) * (1 - alpha_bar_tm1) / (1 - alpha_bar_t)
                x_t = w_x0 * eps_pred + w_xt * x_t + sigma_t * z
            else:
                scaled_noise = sigma_t / torch.sqrt(1 - alpha_bar_t) * eps_pred
                x_t = (x_t - scaled_noise) / torch.sqrt(alpha_t) + sigma_t * z

            x_t_d = torch.clip(x_t[..., :3], min=-1 * clip_bounds_m, max=clip_bounds_m)
            x_t_m = torch.clip(x_t[..., 3:], min=-1 * clip_bounds_d, max=clip_bounds_d)
            x_t = torch.cat((x_t_d, x_t_m), dim=-1)

            if visualize:
                x_ts.append(x_t.detach().clone())

    # For visualization purposes, tack gt onto end
    if visualize:
        return x_t, x_ts, all_pred, noise_samples
    return x_t
