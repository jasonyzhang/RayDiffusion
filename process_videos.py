import os.path as osp
import subprocess
from glob import glob


INPUT_DIR = "resources/co3d_results"
OUTPUT_DIR = "resources/co3d_results_processed"


def process_images(input_dir, output_dir):
    for image in glob(osp.join(input_dir, "*.jpg")):
        output_image = osp.join(output_dir, osp.basename(image).replace(".png", ".jpg"))
        cmd = ["convert", image, "-resize", "400x", output_image]
        print(" ".join(cmd))
        subprocess.call(cmd)


def process_videos(input_dir, output_dir):
    for video in glob(osp.join(input_dir, "*.mp4")):
        output_video = osp.join(output_dir, osp.basename(video))
        cmd = [
            "ffmpeg",
            "-i",
            video,
            "-c:v",
            "h264",
            "-an",
            "-vf",
            "setpts=(2/3)*PTS,scale=400:400",
            output_video,
            "-y",
        ]
        print(" ".join(cmd))
        subprocess.call(cmd)


if __name__ == "__main__":
    process_images(INPUT_DIR, OUTPUT_DIR)
    # process_videos(INPUT_DIR, OUTPUT_DIR)
