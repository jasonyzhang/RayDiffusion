## Training Directions

### Prepare CO3D Dataset

Download the CO3Dv2 Dataset from [here](https://github.com/facebookresearch/co3d/tree/main).

Then, pre-process the annotations:
```
python preprocess_co3d.py --category all --precompute_bbox --co3d_v2_dir /path/to/co3d_v2
python preprocess_co3d.py --category all --co3d_v2_dir /path/to/co3d_v2
```

### Setting up `accelerate`

Use `accelerate config` to set up `accelerate`. We recommend using single GPU without
any mixed precision (we handle AMP ourselves).

### Training models

To train the ray diffusion model, run:
```
accelerate launch --multi_gpu --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8 train.py \
    training.batch_size=8 training.max_iterations=450000
```

To train the ray regression model, run:
```
accelerate launch --multi_gpu --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8 train.py \
    training.batch_size=8 training.max_iterations=300000 training.regression=True
```

Some notes:
* `batch_size` refers to the batch size per GPU. Total batch size will be `batch_size * num_gpu`.
* We train our models on 4 A6000s with a total batch size of 64. You can adjust the number of GPUs and batch size depending on your setup. You may need to adjust the number of training iterations accordingly.
* You can resume training from a checkpoint by specifying `train.resume=True hydra.run.dir=/path/to/output_dir`
* If you are getting NaNs, try turning off mixed precision. This will increase the amount of memory used.

For debugging, we recommend using a single-GPU job with a single category:
```
accelerate launch train.py training.batch_size=4 dataset.category=apple debug.wandb=False hydra.run.dir=output_debug
``` 
