"""Make silent GIF previews and an H.264 browser copy of the supplied videos."""
from pathlib import Path
import argparse
import subprocess
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=ROOT / "assets/videos")
    args = parser.parse_args()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for name in ("sim_grasp", "real_grasp"):
        source = args.video_dir / f"{name}.mp4"
        if not source.is_file():
            raise FileNotFoundError(source)
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                        "-filter_complex", "fps=8,scale=480:-1:flags=lanczos,split[s0][s1];"
                        "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
                        "-an", "-loop", "0", str(args.video_dir / f"{name}_preview.gif")], check=True)
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", "1", "-i", str(source),
                        "-frames:v", "1", str(args.video_dir / f"{name}_poster.png")], check=True)
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i",
                    str(args.video_dir / "real_grasp.mp4"), "-c:v", "libx264", "-crf", "24",
                    "-preset", "medium", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
                    str(args.video_dir / "real_grasp_web.mp4")], check=True)
    print("Saved full-duration, normal-speed silent previews and H.264 real-robot copy; originals retained.")


if __name__ == "__main__":
    main()
