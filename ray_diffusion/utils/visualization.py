import io
import os
import os.path as osp

import ipdb  # noqa: F401
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene

from ray_diffusion.inference.ddpm import inference_ddpm
from ray_diffusion.utils.rays import (
    Rays,
    cameras_to_rays,
    rays_to_cameras,
    rays_to_cameras_homography,
)

cmap = plt.get_cmap("hsv")


def unnormalize_image(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    image = image * std + mean
    return (image * 255.0).astype(np.uint8)


def plot_to_image(figure, dpi=100):
    """Converts matplotlib fig to a png for logging with tf.summary.image."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="raw", dpi=dpi)
    plt.close(figure)
    buffer.seek(0)
    image = np.reshape(
        np.frombuffer(buffer.getvalue(), dtype=np.uint8),
        newshape=(int(figure.bbox.bounds[3]), int(figure.bbox.bounds[2]), -1),
    )
    return image[..., :3]


def view_color_coded_images_from_tensor(images):
    num_frames = images.shape[0]
    num_rows = 2
    num_cols = 4
    figsize = (num_cols * 2, num_rows * 2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()
    for i in range(num_rows * num_cols):
        if i < num_frames:
            axs[i].imshow(unnormalize_image(images[i]))
            for s in ["bottom", "top", "left", "right"]:
                axs[i].spines[s].set_color(cmap(i / (num_frames)))
                axs[i].spines[s].set_linewidth(5)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        else:
            axs[i].axis("off")
    plt.tight_layout()


def view_color_coded_images_from_path(image_dir):
    num_rows = 2
    num_cols = 4
    figsize = (num_cols * 2, num_rows * 2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()

    def hidden(x):
        return not x.startswith(".")

    image_paths = sorted(os.listdir(image_dir))
    image_paths = list(filter(hidden, image_paths))
    image_paths = image_paths[0 : (min(len(image_paths), 8))]
    num_frames = len(image_paths)

    for i in range(num_rows * num_cols):
        if i < num_frames:
            img = np.asarray(Image.open(osp.join(image_dir, image_paths[i])))
            print(img.shape)
            axs[i].imshow(img)
            for s in ["bottom", "top", "left", "right"]:
                axs[i].spines[s].set_color(cmap(i / (num_frames)))
                axs[i].spines[s].set_linewidth(5)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        else:
            axs[i].axis("off")
    plt.tight_layout()
    return fig, num_frames


def create_training_visualizations(
    model,
    images,
    device,
    cameras_gt,
    num_images,
    crop_parameters,
    pred_x0=False,
    moments_rescale=1.0,
    visualize_pred=False,
    return_first=False,
    calculate_intrinsics=False,
):
    num_patches_x = num_patches_y = model.width
    rays_final, rays_intermediate, pred_intermediate, _ = inference_ddpm(
        model,
        images,
        device,
        visualize=True,
        clip_bounds_d=5,
        clip_bounds_m=5,
        pred_x0=pred_x0,
        crop_parameters=crop_parameters,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
    )

    T = model.noise_scheduler.max_timesteps
    if T == 1000:
        ts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
    else:
        ts = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

    # Get predicted cameras from rays
    pred_cameras_batched = []
    vis_images = []
    for index in range(len(images)):
        pred_cameras = []
        per_sample_images = []
        for ii in range(num_images):
            rays_gt = cameras_to_rays(
                cameras_gt[index],
                crop_parameters[index],
                num_patches_x=num_patches_x,
                num_patches_y=num_patches_y,
            )
            image_vis = (images[index, ii].cpu().permute(1, 2, 0).numpy() + 1) / 2
            fig, axs = plt.subplots(3, 9, figsize=(12, 4.5), dpi=100)

            for i, t in enumerate(ts):
                r, c = i // 4, i % 4
                if visualize_pred:
                    curr = pred_intermediate[t][index]
                else:
                    curr = rays_intermediate[t][index]
                rays = Rays.from_spatial(curr, moments_rescale=moments_rescale)
                vis = (
                    torch.nn.functional.normalize(rays.get_moments()[ii], dim=-1) + 1
                ) / 2
                axs[r, c].imshow(vis.reshape(num_patches_y, num_patches_x, 3).cpu())
                axs[r, c].set_title(f"T={T - t}")

            i += 1
            r, c = i // 4, i % 4

            vis = (
                torch.nn.functional.normalize(rays_gt.get_moments()[ii], dim=-1) + 1
            ) / 2
            axs[r, c].imshow(vis.reshape(num_patches_y, num_patches_x, 3).cpu())
            axs[r, c].set_title("GT Moments")

            for i, t in enumerate(ts):
                r, c = i // 4, i % 4 + 4
                if visualize_pred:
                    curr = pred_intermediate[t][index]
                else:
                    curr = rays_intermediate[t][index]
                rays = Rays.from_spatial(curr, moments_rescale=moments_rescale)
                vis = (
                    torch.nn.functional.normalize(rays.get_directions()[ii], dim=-1) + 1
                ) / 2
                axs[r, c].imshow(vis.reshape(num_patches_y, num_patches_x, 3).cpu())
                axs[r, c].set_title(f"T={T - t}")

            i += 1
            r, c = i // 4, i % 4 + 4
            vis = (
                torch.nn.functional.normalize(rays_gt.get_directions()[ii], dim=-1) + 1
            ) / 2
            axs[r, c].imshow(vis.reshape(num_patches_y, num_patches_x, 3).cpu())
            axs[r, c].set_title("GT Directions")

            axs[2, -1].imshow(image_vis)
            axs[2, -1].set_title("Input Image")
            for s in ["bottom", "top", "left", "right"]:
                axs[2, -1].spines[s].set_color(cmap(ii / (num_images)))
                axs[2, -1].spines[s].set_linewidth(5)

            for ax in axs.flatten():
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            img = plot_to_image(fig)
            plt.close()
            per_sample_images.append(img)

            if return_first:
                rays_camera = pred_intermediate[0][index]
            elif pred_x0:
                rays_camera = pred_intermediate[-1][index]
            else:
                rays_camera = rays_final[index]
            rays = Rays.from_spatial(rays_camera, moments_rescale=moments_rescale)

            if calculate_intrinsics:
                pred_camera = rays_to_cameras_homography(
                    rays=rays[ii, None],
                    crop_parameters=crop_parameters[index],
                    num_patches_x=num_patches_x,
                    num_patches_y=num_patches_y,
                )
            else:
                pred_camera = rays_to_cameras(
                    rays=rays[ii, None],
                    crop_parameters=crop_parameters[index],
                    num_patches_x=num_patches_x,
                    num_patches_y=num_patches_y,
                )
            pred_cameras.append(pred_camera[0])
        pred_cameras_batched.append(pred_cameras)
        vis_images.append(np.vstack(per_sample_images))

    return vis_images, pred_cameras_batched


def create_plotly_cameras_visualization(cameras_gt, cameras_pred, num):
    num_frames = cameras_gt.R.shape[0]
    name = f"Vis {num} GT vs Pred Cameras"
    camera_scale = 0.05

    # Cameras_pred is already a 2D list of unbatched cameras
    # But cameras_gt is a 1D list of batched cameras
    scenes = {f"Vis {num} GT vs Pred Cameras": {}}
    for i in range(num_frames):
        scenes[name][f"Pred Camera {i}"] = PerspectiveCameras(
            R=cameras_pred[i].R, T=cameras_pred[i].T
        )
    for i in range(num_frames):
        scenes[name][f"GT Camera {i}"] = PerspectiveCameras(
            R=cameras_gt.R[i].unsqueeze(0), T=cameras_gt.T[i].unsqueeze(0)
        )

    fig = plot_scene(
        scenes,
        camera_scale=camera_scale,
    )
    fig.update_scenes(aspectmode="data")
    fig.update_layout(height=800, width=800)

    for i in range(num_frames):
        fig.data[i].line.color = matplotlib.colors.to_hex(cmap(i / (num_frames)))
        fig.data[i].line.width = 4
        fig.data[i + num_frames].line.dash = "dash"
        fig.data[i + num_frames].line.color = matplotlib.colors.to_hex(
            cmap(i / (num_frames))
        )
        fig.data[i + num_frames].line.width = 4

    return fig
