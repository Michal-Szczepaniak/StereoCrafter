"""Run only the depth-splatting (forward-warp) stage, from a DepthCrafter
depth checkpoint produced elsewhere.

Pairs with depth_splatting_inference.py's --depth_only=True: run that on the
GPU box, fetch its {output_dir}/.depth_checkpoint/ directory (manifest.json +
the quantized depth chunk .mkv files) plus the source video, then point this
script at the fetched checkpoint dir to iterate on splatting parameters
(max_disp, disp_tolerance) without touching a GPU.

manifest.json already has everything depth-pass-specific (chunk file names,
per-chunk quantization range, global min/max, and the run's own params dict).
The only things DepthSplatting also needs - original_height/width, stride,
target_fps - are pure functions of (source video, max_res, target_fps,
dataset) computed by get_video_info(), which does no GPU work (decord only) -
so they're recomputed here instead of needing to be persisted separately.

Usage:
    python splat_from_checkpoint.py \\
        --checkpoint_dir /path/to/fetched/.depth_checkpoint \\
        --input_video_path /path/to/same/source.mp4 \\
        --store_dir ./outputs/experiment/splat \\
        --max_disp 20.0 --disp_tolerance 1.0 --minutes 5
"""

import json
import os

from fire import Fire

from depth_splatting_inference import DepthSplatting, get_video_info


def main(
    checkpoint_dir: str,
    input_video_path: str,
    store_dir: str,
    max_disp: float = 20.0,
    batch_size: int = 10,
    disp_tolerance: float = 1.0,
    compress_store: bool = True,
    device: str = "cpu",
    dataset: str = "open",
    keep_depth_chunks: bool = True,
    minutes: float = None,
):
    """device: "cpu" by default (the point of this script) - pass "cuda" to
    run it on a GPU machine too, e.g. to A/B splatting params quickly
    without re-running the depth pass.

    keep_depth_chunks: True by default (opposite of DepthSplatting's own
    default) - a fetched checkpoint is meant to be reused across multiple
    splatting attempts, not consumed by the first one. Pass False if this
    checkpoint is disposable and you want it cleaned up as it's consumed.

    minutes: None (default) splats the whole checkpoint. Set to stop after
    this many minutes of output instead - e.g. --minutes 5 for a quick test
    without waiting on (or consuming) the rest of the episode. Converted to
    a frame count using target_fps once it's known (see get_video_info call
    below), then passed through as DepthSplatting's max_frames.
    """
    manifest_path = os.path.join(checkpoint_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    params = manifest["params"]
    chunk_files = [os.path.join(checkpoint_dir, name) for name in manifest["chunk_files"]]
    chunk_meta = manifest["chunk_meta"]
    global_min = manifest["global_min"]
    global_max = manifest["global_max"]

    (
        _original_num_frames,
        original_height,
        original_width,
        _source_fps,
        target_fps,
        stride,
        _processing_height,
        _processing_width,
        _total_frames,
    ) = get_video_info(input_video_path, params["max_res"], params["target_fps"], dataset)

    max_frames = round(minutes * 60 * target_fps) if minutes else None

    DepthSplatting(
        input_video_path,
        store_dir,
        chunk_files,
        chunk_meta,
        global_min,
        global_max,
        max_disp,
        batch_size,
        target_fps,
        stride,
        original_height,
        original_width,
        store_params={**params, "max_disp": max_disp, "disp_tolerance": disp_tolerance},
        compress_store=compress_store,
        disp_tolerance=disp_tolerance,
        device=device,
        keep_depth_chunks=keep_depth_chunks,
        max_frames=max_frames,
    )


if __name__ == "__main__":
    Fire(main)
