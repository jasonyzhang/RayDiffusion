## Evaluation Directions

Use the scripts from ray_diffusion/eval to evaluate the performance of the model on the
CO3D dataset:

```
python -m ray_diffusion.eval.eval_jobs --eval_type diffusion --eval_path models/co3d_diffusion
python -m ray_diffusion.eval.eval_jobs --eval_type regression --eval_path models/co3d_regression
```

The expected output at the end of evaluating the diffusion model is:
```
N=            2     3     4     5     6     7     8
Seen R    0.918 0.924 0.926 0.929 0.931 0.933 0.933
Seen CC   1.000 0.942 0.905 0.878 0.862 0.850 0.841
Unseen R  0.835 0.856 0.863 0.869 0.872 0.875 0.881
Unseen CC 1.000 0.877 0.811 0.770 0.741 0.724 0.714
```
This reports the rotation and camera center accuracy on held out sequences on both seen and unseen object categories
for different numbers of images. We average performance over 5 runs to reduce variance.

Note that there may be some minor differences in the numbers due to randomness in the evaluation
and inference processes.

The evaluation scripts will take a while to run. It may be preferable to parallelize the script
using submitit.