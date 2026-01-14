import os
import subprocess

ROOT = "data/raw/fsl105/clips"

for root, _, files in os.walk(ROOT):
    for file in files:
        if file.lower().endswith(".mov"):
            src = os.path.join(root, file)

            dst = os.path.splitext(src)[0] + ".mp4"

            if os.path.exists(dst):
                continue

            print("Converting:", src)

            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", src,
                    "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p",
                    dst
                ],
                check=True
            )
