# Ray Diffusion


## Setting up Environment

We recommend using a conda environment to manage dependencies. Install a version of
Pytorch compatible with your CUDA version from the [Pytorch website](https://pytorch.org/get-started/locally/).

```
conda create -n raydiffusion python=3.10
conda activate raydiffusion
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install xformers -c xformers
pip install -r requirements.txt
```

Then, follow the directions to install Pytorch3D [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).


## Run Demo

Run ray diffusion with known bounding boxes (provided as a json):
```
python demo.py  --model_dir models/co3d_diffusion --image_dir examples/robot/images \
    --bbox_path examples/robot/bboxes.json --output_path robot.html
```

Run ray diffusion with bounding boxes extracted automatically from masks:
```
python demo.py  --model_dir models/co3d_diffusion --image_dir examples/robot/images \
    --mask_dir examples/robot/masks --output_path robot.html
```

Run ray regression:
```
python demo.py  --model_dir models/co3d_regression --image_dir examples/robot/images \
    --bbox_path examples/robot/bboxes.json --output_path robot.html
```

## Citing Cameras as Rays

If you find this code helpful, please cite:

```
@InProceedings{zhang2024raydiffusion,
    title={Cameras as Rays: Sparse-view Pose Estimation via Ray Diffusion},
    author={Zhang, Jason Y and Lin, Amy and Kumar, Moneish and Yang, Tzu-Hsuan and Ramanan, Deva and Tulsiani, Shubham},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2024}
}
```