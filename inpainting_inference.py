import os
import gc
import json
import shutil
import subprocess
import sys
import time
import cv2
import numpy as np
from fire import Fire

import torch
import torch.nn.functional as F

from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder
from diffusers import UNetSpatioTemporalConditionModel

from pipelines.stereo_video_inpainting import StableVideoDiffusionInpaintingPipeline, tensor2vid
from splat_store import open_store
from chunked_attention import enable_chunked_attention


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


class LiveProgress:
    """One continuously-updating status line covering both levels of
    progress this stage has (outer chunk/ETA, and the per-tile denoising
    step) instead of three separate noisy streams - the outer "Process:
    i/N" print, per-tile "TILE [i,j]" prints, and diffusers' own per-tile
    tqdm bar (0/8, restarting for every one of tile_num^2 tiles). Redraws
    in place via \\r; only commits a real newline once per outer chunk
    (set_header), so nothing scrolls except one line per chunk."""

    def __init__(self, num_frames: int, total_iters: int):
        self.num_frames = num_frames
        self.total_iters = total_iters
        self._header = ""
        self._tile_idx = 0
        self._tile_total = 0
        self._tile_pos = ""
        self._step = 0
        self._num_steps = 0
        self._last_len = 0

    def set_header(self, i, iter_index, overlap, cur_i, cur_overlap, timing_str):
        self._header = (
            f"Process: {i}/{self.num_frames} (chunk {iter_index}/{self.total_iters}), "
            f"overlap {overlap}, cur_i {cur_i} cur_overlap {cur_overlap}{timing_str}"
        )
        self._tile_idx = self._tile_total = self._step = self._num_steps = 0
        self._render()

    def set_tile(self, tile_idx: int, tile_total: int, i: int, j: int):
        self._tile_idx, self._tile_total = tile_idx, tile_total
        self._tile_pos = f"[{i},{j}]"
        self._step = self._num_steps = 0
        self._render()

    def set_step(self, step: int, num_steps: int):
        self._step, self._num_steps = step, num_steps
        self._render()

    def _render(self):
        line = self._header
        if self._tile_total:
            line += f" | tile {self._tile_idx}/{self._tile_total} {self._tile_pos}"
            if self._num_steps:
                line += f" | step {self._step}/{self._num_steps}"
        pad = max(self._last_len - len(line), 0)
        sys.stdout.write("\r" + line + " " * pad)
        sys.stdout.flush()
        self._last_len = len(line)

    def finish_iter(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


def _round_up_to_multiple(x: int, multiple: int = 64) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _prefill_occlusion(warp_np: np.ndarray, mask_np: np.ndarray, radius: int = 5) -> np.ndarray:
    """Classical inpaint (cv2.INPAINT_TELEA) of the occluded holes in the raw
    warp frame before the diffusion model sees it. Diffusion conditioned
    directly on a pure-black hole (esp. a disocclusion column pinned to the
    frame's true edge, where there's no real content on one side to lean on)
    tends to just reproduce the black rather than hallucinate over it, even
    with more steps/guidance - a plausible classical starting point gives it
    something to refine instead of invent from nothing."""
    out = warp_np.copy()
    for t in range(warp_np.shape[0]):
        hole = (mask_np[t] > 127).astype(np.uint8) * 255
        if hole.any():
            out[t] = cv2.inpaint(warp_np[t], hole, radius, cv2.INPAINT_TELEA)
    return out


def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1) / overlap_size).to(b.device)
    b[:, :, :, :overlap_size] = (1 - weight_b) * a[:, :, :, -overlap_size:] + weight_b * b[:, :, :, :overlap_size]
    return b


def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, -1, 1) / overlap_size).to(b.device)
    b[:, :, :overlap_size, :] = (1 - weight_b) * a[:, :, -overlap_size:, :] + weight_b * b[:, :, :overlap_size, :]
    return b


def spatial_tiled_process(
    cond_frames,
    mask_frames,
    process_func,
    tile_num,
    spatial_n_compress=8,
    progress: "LiveProgress | None" = None,
    **kargs,
):
    height = cond_frames.shape[2]
    width = cond_frames.shape[3]

    tile_overlap = (128, 128)
    # NOTE: tile_size must be a multiple of 64, not just 8. The pipeline's own
    # check (stereo_video_inpainting.py's `height % 8 != 0` guard) only
    # enforces what the VAE needs, but the UNet has 3 downsample stages on
    # top of that (see unet/config.json down_block_types), so skip
    # connections need the *latent* dims (pixel / 8) divisible by 8 too -
    # i.e. pixel dims divisible by 64. A plain floor (e.g. 352x576 at
    # 1920x1080) can miss that and crash with a skip-connection size
    # mismatch ("Sizes of tensors must match except in dimension 1").
    # Rounding up to 128 "fixed" that but overshoots - it inflates every
    # interior tile's area by ~20% (e.g. 352x576 -> 384x640), which was
    # enough extra VRAM per forward pass to OOM by the last tile of the
    # first chunk. Round up to 64 instead: the minimum that keeps the shape
    # valid without the extra bloat (e.g. 352x576 -> 384x576, only the
    # dimension that actually needed it grows).
    raw_tile_h = int((height + tile_overlap[0] * (tile_num - 1)) / tile_num)
    raw_tile_w = int((width + tile_overlap[1] * (tile_num - 1)) / tile_num)

    tile_size = (
        ((raw_tile_h + 63) // 64) * 64,
        ((raw_tile_w + 63) // 64) * 64,
    )
    tile_stride = (
        tile_size[0] - tile_overlap[0],
        tile_size[1] - tile_overlap[1],
    )

    # Feeds per-step updates into the single shared status line instead of
    # diffusers' own per-tile tqdm bar (a fresh 0/num_inference_steps bar
    # for every one of tile_num^2 tiles otherwise). One callback instance
    # reused across tiles - it only reports *which step*, not *which
    # tile*, since progress.set_tile() below already tracks that.
    if progress is not None:
        num_steps_for_progress = kargs.get("num_inference_steps")

        def _progress_step_callback(pipe, step, timestep, callback_kwargs):
            progress.set_step(step + 1, num_steps_for_progress)
            return callback_kwargs

        kargs = dict(kargs, callback_on_step_end=_progress_step_callback)

    tile_total = tile_num * tile_num
    tile_counter = 0
    cols = []
    for i in range(0, tile_num):
        rows = []
        for j in range(0, tile_num):
            cond_tile = cond_frames[
                :, :,
                i * tile_stride[0] : i * tile_stride[0] + tile_size[0],
                j * tile_stride[1] : j * tile_stride[1] + tile_size[1],
            ]
            mask_tile = mask_frames[
                :, :,
                i * tile_stride[0] : i * tile_stride[0] + tile_size[0],
                j * tile_stride[1] : j * tile_stride[1] + tile_size[1],
            ]

            tile_counter += 1
            if progress is not None:
                progress.set_tile(tile_counter, tile_total, i, j)
            else:
                print(
                    f"TILE [{i},{j}]: "
                    f"shape={tuple(cond_tile.shape)}, "
                    f"H={cond_tile.shape[2]}, "
                    f"W={cond_tile.shape[3]}"
                )
            tile = process_func(
                frames=cond_tile,
                frames_mask=mask_tile,
                height=cond_tile.shape[2],
                width=cond_tile.shape[3],
                num_frames=len(cond_tile),
                output_type="latent",
                **kargs,
            ).frames[0]

            rows.append(tile)

            # OPTIMIZATION (VRAM): with cpu offload, submodules are moved back to
            # CPU after each tile's forward pass, but the caching allocator can
            # still hold onto fragmented reserved blocks. Reclaim them between
            # tiles rather than only once per frame chunk - higher tile_num means
            # more forward passes per chunk, so fragmentation compounds faster.
            gc.collect()
            torch.cuda.empty_cache()
        cols.append(rows)

    latent_stride = (tile_stride[0] // spatial_n_compress, tile_stride[1] // spatial_n_compress)
    latent_overlap = (tile_overlap[0] // spatial_n_compress, tile_overlap[1] // spatial_n_compress)

    results_cols = []
    for i, rows in enumerate(cols):
        results_rows = []
        for j, tile in enumerate(rows):
            if i > 0:
                tile = blend_v(cols[i - 1][j], tile, latent_overlap[0])
            if j > 0:
                tile = blend_h(rows[j - 1], tile, latent_overlap[1])
            results_rows.append(tile)
        results_cols.append(results_rows)

    pixels = []
    for i, rows in enumerate(results_cols):
        for j, tile in enumerate(rows):
            if i < len(results_cols) - 1:
                tile = tile[:, :, : latent_stride[0], :]
            if j < len(rows) - 1:
                tile = tile[:, :, :, : latent_stride[1]]
            rows[j] = tile
        pixels.append(torch.cat(rows, dim=3))
    x = torch.cat(pixels, dim=2)

    # OPTIMIZATION: tiles are no longer needed once merged - drop refs so the
    # (potentially large, tile_num>1) intermediate tile tensors get freed now
    # instead of lingering until the caller's scope exits.
    del cols, results_cols, pixels
    return x


def _to_uint8_rgb(frame_float_chw: torch.Tensor) -> np.ndarray:
    """[t, c, h, w] float32 in [0,1] -> [t, h, w, c] uint8, no extra copies held."""
    arr = (frame_float_chw.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()
    return arr


class FFmpegSegmentWriter:
    """Pipes raw RGB frames straight into an ffmpeg subprocess encoding one
    chunk to its own standalone .mkv segment (see _concat_segments below for
    how multiple segments get losslessly stitched into the final output).

    Replaces the old cv2.VideoWriter(fourcc mp4v, .mp4) writer. mp4 requires
    a moov atom written at the very end to be a valid/seekable file - if the
    process dies mid-write the file is generally unusable. matroska (.mkv)
    has no such finalization step: everything muxed so far is valid and
    playable even if the process is killed right after, which is what makes
    both live preview (open the segment in a player while it's still being
    written) and this module's resumable checkpointing (see main()) work.

    NOTE: ffmpeg's matroska muxer has no way to set timestamp granularity
    finer than 1ms (-video_track_timescale and -enc_time_base are both
    no-ops for it, verified empirically) - unlike mp4's exact
    frame-rate-derived timebase (see the old comment this replaced, in git
    history). That puts a small (~1ms/frame) PTS quantization jitter into
    each segment. It's harmless here: the stage-3 combine command already
    forces both eyes through an explicit `fps=` filter before hstack, which
    re-times both streams to a common rate regardless of source-container
    jitter - verified on a synthetic test (one exact-timebase mp4 input +
    one 1ms-jittery mkv input through that exact filter): 480/480 frames,
    zero drops/dupes, uniform output timing.
    """

    def __init__(self, path, fps, width, height, crf=16):
        self.path = path
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-f", "matroska",
            path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frames_rgb_uint8: np.ndarray):
        # frames_rgb_uint8: [t, h, w, c], RGB order
        self.proc.stdin.write(frames_rgb_uint8.tobytes())

    def release(self):
        self.proc.stdin.close()
        ret = self.proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with code {ret} while writing {self.path}")


def _concat_segments(segment_paths, output_path, fps, width, height):
    """Stitch segments (all encoded with identical settings by
    FFmpegSegmentWriter) into a single output file.

    NOT ffmpeg's concat demuxer + stream copy (`-f concat -safe 0 -i list.txt
    -c copy`) - that spliced compressed bitstreams from ~300 independently
    keyframed tiny segments (2-5 frames each) and produced a genuinely
    invalid file: verified via `ffmpeg -i right.mkv -vsync 0 out_%04d.png`
    logging "non monotonically increasing dts to muxer" at 7 evenly-spaced
    points in a real 604-frame/301-segment run. That's not cosmetic -
    different decode paths handled the corruption differently (some
    silently dropped or reordered a frame right at that point, some didn't),
    which was the actual root cause of a user-reported permanent one-frame
    skew between the two eyes from that point in the video onward (traced
    all the way back from the final SBS output, through hstack/hstack-only
    raw-pixel combines, to this file itself - stage 1's warp/mask output
    was verified frame-for-frame correct throughout, so the corruption is
    introduced here).

    Instead: decode every segment to raw RGB (no compressed bitstream, no
    per-segment GOP/keyframe structure, nothing for a timestamp to get
    wrong) and pipe the concatenated byte stream through a single fresh
    encode - same principle as FFmpegSegmentWriter's own raw-pixel writing,
    just applied once more at the joining step instead of trusting container
    timestamps to survive a compressed-stream splice.
    """
    writer_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-f", "matroska",
        output_path,
    ]
    writer = subprocess.Popen(writer_cmd, stdin=subprocess.PIPE)
    for seg_path in segment_paths:
        reader = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", seg_path,
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, check=True,
        )
        writer.stdin.write(reader.stdout)
    writer.stdin.close()
    ret = writer.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg exited with code {ret} while concatenating segments into {output_path}")


def main(
    pre_trained_path,
    unet_path,
    splat_store_dir,
    save_dir,
    frames_chunk=23,
    overlap=3,
    tile_num=1,
    decode_chunk_size=8,
    decode_latents_chunk_size=2,
    vae_encode_chunk_size=5,
    enable_model_cpu_offload=False,
    enable_sequential_cpu_offload=False,
    enable_vae_slicing=True,
    enable_vae_tiling=False,
    num_inference_steps=8,
    min_guidance_scale=1.01,
    max_guidance_scale=1.01,
    noise_aug_strength=0.0,
    max_iters=None,
    prefill_occlusion=True,
    resume=True,
    chunked_attention=True,
    attention_kv_chunk_size=1024,
    suppress_attention_kernel_warnings=True,
):
    """NOTE: --input_video_path is now --splat_store_dir - point it at the
    directory depth_splatting_inference.py wrote (warp.npy + mask.npy +
    meta.json), not an mp4.

    This stage no longer touches the original video at all, or writes SBS /
    anaglyph composites - it only ever reads warp.npy/mask.npy and produces
    the generated right-eye video. Nothing here needs the source video: the
    diffusion pipeline is only ever conditioned on warp+mask, never on
    "left" - that was only read before to build the SBS/anaglyph previews in
    this same process. Combine the right-eye output with the original video
    afterwards (ffmpeg command printed at the end of the run) - cheaper to
    redo if you ever want a different composite, and keeps this GPU-heavy
    stage from paying for video decode + two extra encodes it doesn't need.

    resume: continue a previous run's checkpoint under
    {save_dir}/.inpaint_checkpoint_{video_name} if its parameters match this
    run's - each processed chunk is written to its own .mkv segment and
    recorded in the checkpoint as soon as it's done, so a crash only costs
    the in-flight chunk. Segments are losslessly concatenated into the
    final {video_name}_right.mkv once the whole video is done, and the
    checkpoint is deleted at that point. Set resume=False to force a clean
    restart. A run stopped early via max_iters is resumable the same way -
    the printed combine commands are skipped until a run actually finishes.

    chunked_attention: this GPU's ROCm build has no working flash/efficient
    SDPA kernel (confirmed empirically - see chunked_attention.py's module
    docstring), so PyTorch's naive "math" attention fallback is used, which
    needs O(seq_len^2) memory - fine for small tiles, but explosive for
    bigger ones (tile_num=2 needs ~20GB for a single attention call at
    1080p, independent of how much VRAM the card actually has). Leave this
    on to use a chunked/online-softmax attention implementation instead
    (O(seq_len) memory, some compute overhead) - lets tile_num go lower
    (bigger tiles, fewer of them) on any card. attention_kv_chunk_size
    trades memory for speed: smaller = less VRAM, more (smaller) matmuls.
    suppress_attention_kernel_warnings: chunked_attention always probes for
    a real flash/efficient kernel first (see chunked_attention.py) - on
    hardware where that probe always fails (this GPU), PyTorch logs a
    UserWarning per unavailable backend on every single call, which spams
    the LiveProgress status line. On by default; set False to see them
    again (e.g. to confirm the fast path is engaging on different hardware).
    """
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pre_trained_path, subfolder="image_encoder", variant="fp16", torch_dtype=torch.float16
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pre_trained_path, subfolder="vae", variant="fp16", torch_dtype=torch.float16
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        unet_path, subfolder="unet_diffusers", low_cpu_mem_usage=True, torch_dtype=torch.float16
    )

    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    pipeline = StableVideoDiffusionInpaintingPipeline.from_pretrained(
        pre_trained_path,
        image_encoder=image_encoder,
        vae=vae,
        unet=unet,
        torch_dtype=torch.float16,
    )

    # diffusers' own per-call tqdm bar would otherwise print a fresh
    # 0/num_inference_steps bar for every one of tile_num^2 tiles - replaced
    # by LiveProgress's single status line (fed via callback_on_step_end,
    # see spatial_tiled_process) instead.
    pipeline.set_progress_bar_config(disable=True)

    # OPTIMIZATION (VRAM): these are cheap opt-ins if you're still tight on GPU
    # memory after the RAM fix below. cpu offload trades speed for VRAM.
    if enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to("cuda")
    if enable_vae_slicing and hasattr(pipeline.vae, "enable_slicing"):
        pipeline.vae.enable_slicing()
    if enable_vae_tiling and hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()
    if chunked_attention:
        enable_chunked_attention(
            pipeline.unet, kv_chunk_size=attention_kv_chunk_size,
            suppress_probe_warnings=suppress_attention_kernel_warnings,
        )
        enable_chunked_attention(
            pipeline.vae, kv_chunk_size=attention_kv_chunk_size,
            suppress_probe_warnings=suppress_attention_kernel_warnings,
        )

    os.makedirs(save_dir, exist_ok=True)

    # warp_store / mask_store are numpy memmaps: indexing a slice touches only
    # that slice's pages on disk, never the whole array (num_frames could be
    # thousands of frames - this is the same "no full video in RAM" guarantee
    # a chunked video reader had, just without a video codec, and without
    # needing to read the source video at all anymore).
    warp_store, mask_store, meta = open_store(splat_store_dir, mode="r")
    num_frames = warp_store.shape[0]
    # height/width: the REAL output resolution - every pixel of the source
    # frame, never cropped.
    height, width = warp_store.shape[1], warp_store.shape[2]
    # padded_height/padded_width: the canvas the model actually runs on.
    # UNet/VAE need dims divisible by 64 (see spatial_tiled_process's
    # tile-size comment), and 1080 (etc.) usually isn't - so pad up to the
    # nearest multiple of 64 with edge-replicated rows/cols instead of
    # flooring down and losing real content. The padding is stripped back
    # off right before each chunk is written (see generated_out below), so
    # it never reaches the output file.
    padded_height = _round_up_to_multiple(height)
    padded_width = _round_up_to_multiple(width)
    pad_bottom = padded_height - height
    pad_right = padded_width - width

    fps = meta["fps"]
    store_name = os.path.basename(os.path.normpath(splat_store_dir))
    video_name = store_name + "_inpainting_results"

    # ------------------------------------------------------------------
    # Checkpoint bootstrap: resume a previous run's progress (one .mkv
    # segment per processed chunk, see FFmpegSegmentWriter) if the params
    # match, else start fresh. See main()'s docstring.
    # ------------------------------------------------------------------
    checkpoint_dir = os.path.join(save_dir, f".inpaint_checkpoint_{video_name}")
    manifest_path = os.path.join(checkpoint_dir, "manifest.json")
    tail_path = os.path.join(checkpoint_dir, "generated_tail.pt")

    run_params = {
        "splat_store_dir": os.path.abspath(splat_store_dir),
        "frames_chunk": frames_chunk,
        "overlap": overlap,
        "tile_num": tile_num,
        "num_inference_steps": num_inference_steps,
        "min_guidance_scale": min_guidance_scale,
        "max_guidance_scale": max_guidance_scale,
        "noise_aug_strength": noise_aug_strength,
        "prefill_occlusion": prefill_occlusion,
        "vae_encode_chunk_size": vae_encode_chunk_size,
        "num_frames": num_frames,
        "width": width,
        "height": height,
    }

    segments = []
    next_i = 0
    generated = None

    if not resume:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    elif os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("params") == run_params and manifest.get("segments"):
            segments = manifest["segments"]
            next_i = manifest["next_i"]
            if next_i > 0 and os.path.exists(tail_path):
                generated = torch.load(tail_path, map_location="cuda")
            print(
                f"==> RESUMING inpainting from checkpoint: {len(segments)} "
                f"chunk(s) already done, continuing at frame {next_i}"
            )
        else:
            print("==> checkpoint found but parameters differ from this run - starting fresh")
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Any segment file left over from a run that crashed mid-chunk (never
    # got cleanly appended to `segments` + recorded in the manifest) is
    # stale - that chunk's work just gets redone, simpler and safer than
    # trying to determine how many of its frames actually made it to disk
    # before the process died.
    for fname in os.listdir(checkpoint_dir):
        if fname.endswith(".mkv") and fname not in segments:
            os.remove(os.path.join(checkpoint_dir, fname))

    def save_checkpoint():
        tmp_path = manifest_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"params": run_params, "next_i": next_i, "segments": segments}, f)
        os.replace(tmp_path, manifest_path)

    def read_chunk(start: int, end: int):
        """Read only [start, end) frames from the memmap store, padded up to
        (padded_height, padded_width) with replicated edge pixels. Never
        touches frames outside this window, and never touches the original
        video."""
        end = min(end, num_frames)

        # np.asarray() on a memmap slice is what actually faults in those
        # pages from disk - only [start:end], nothing else.
        warp_np = np.array(warp_store[start:end], copy=True)
        mask_np = np.array(mask_store[start:end], copy=True)

        if prefill_occlusion:
            warp_np = _prefill_occlusion(warp_np, mask_np)

        warp = torch.from_numpy(warp_np).permute(0, 3, 1, 2).float() / 255.0
        del warp_np

        mask = torch.from_numpy(mask_np).float().unsqueeze(1) / 255.0
        del mask_np

        if pad_bottom or pad_right:
            warp = F.pad(warp, (0, pad_right, 0, pad_bottom), mode="replicate")
            mask = F.pad(mask, (0, pad_right, 0, pad_bottom), mode="replicate")

        return mask, warp

    # NOTE: `generated` was already set above during checkpoint bootstrap -
    # either the resumed tail tensor, or None for a fresh start.
    # OPTIMIZATION: rough total iteration count and a rolling average of
    # per-iteration wall time, just to print an ETA - the last iteration's
    # step size can differ (see cur_i clamping below), so this is an
    # estimate, not an exact countdown.
    step = max(frames_chunk - overlap, 1)
    total_iters = max((num_frames - overlap + step - 1) // step, 1)
    iter_index = 0
    iter_durations = []
    iter_start = None
    completed = False
    progress = LiveProgress(num_frames, total_iters)
    with torch.inference_mode():  # OPTIMIZATION: skip autograd bookkeeping entirely
        i = next_i
        while True:
            if i + overlap >= num_frames:
                completed = True
                break

            if iter_start is not None:
                iter_durations.append(time.monotonic() - iter_start)
            iter_start = time.monotonic()

            if generated is not None and i + frames_chunk > num_frames:
                cur_i = max(num_frames + overlap - frames_chunk, 0)
                cur_overlap = i - cur_i + overlap
            else:
                cur_i = i
                cur_overlap = overlap

            mask_i, warp_i = read_chunk(cur_i, cur_i + frames_chunk)
            input_frames_i = warp_i  # warp_i is already a fresh clone, safe to mutate

            iter_index += 1
            if iter_durations:
                avg_iter_time = sum(iter_durations) / len(iter_durations)
                last_iter_time = iter_durations[-1]
                remaining_iters = max(total_iters - iter_index + 1, 0)
                eta = _format_duration(avg_iter_time * remaining_iters)
                timing_str = (
                    f", last_iter {last_iter_time:.1f}s, avg_iter {avg_iter_time:.1f}s, "
                    f"ETA {eta}"
                )
            else:
                timing_str = ""
            progress.set_header(i, iter_index, overlap, cur_i, cur_overlap, timing_str)

            if generated is not None:
                try:
                    input_frames_i[:cur_overlap] = generated[-cur_overlap:]
                except Exception as e:
                    print(e)
                    print(
                        f"i: {i}, cur_i: {cur_i}, cur_overlap: {cur_overlap}, "
                        f"input_frames_i: {input_frames_i.shape}, generated: {generated.shape}"
                    )

            video_latents = spatial_tiled_process(
                input_frames_i,
                mask_i,
                pipeline,
                tile_num,
                spatial_n_compress=8,
                progress=progress,
                min_guidance_scale=min_guidance_scale,
                max_guidance_scale=max_guidance_scale,
                decode_chunk_size=decode_chunk_size,
                vae_encode_chunk_size=vae_encode_chunk_size,
                fps=7,
                motion_bucket_id=127,
                noise_aug_strength=noise_aug_strength,
                num_inference_steps=num_inference_steps,
            )
            video_latents = video_latents.unsqueeze(0)

            # NOTE: original code had `if video_latents == torch.float16:` here,
            # comparing a Tensor to a dtype object - that's always False and the
            # intended cast never ran. vae is already loaded fp16, so this was a
            # silent no-op; removed rather than "fixed" since it wasn't doing
            # anything useful even when working.

            decoded_frames = []

            for decoded_chunk in pipeline.decode_latents_streaming(
                video_latents,
                num_frames=video_latents.shape[1],
                decode_chunk_size=decode_latents_chunk_size,
            ):
                # decode_latents_streaming yields the VAE decoder's raw
                # output, which is in ~[-1, 1] - not [0, 1]. The old code
                # path went through tensor2vid()/VaeImageProcessor.postprocess,
                # which denormalizes with (x / 2 + 0.5) before clamping; this
                # streaming path skipped that, so clamping straight to [0, 1]
                # was crushing every negative value to black and clipping the
                # rest, which read as blown-out/cranked color and clipped-edge
                # noise in the output.
                decoded_chunk = (decoded_chunk / 2 + 0.5).clamp(0, 1)

                # [frames, channels, height, width]
                for frame in decoded_chunk:
                    decoded_frames.append(frame)

            generated = torch.stack(decoded_frames)

            # cpu offload chains image_encoder->unet->vae: each stage is only
            # offloaded back to CPU when the *next* stage's forward starts, so
            # vae (last in the chain) would otherwise sit on GPU until the
            # chain wraps around at the next chunk's first tile - overlapping
            # with image_encoder/unet onloading right at that boundary. Force
            # it off proactively instead of waiting for that handshake.
            if enable_model_cpu_offload or enable_sequential_cpu_offload:
                pipeline.vae.to("cpu")

            # OPTIMIZATION (VRAM): free GPU-side intermediates as soon as we're
            # done with them instead of waiting for the next loop iteration's
            # allocations to trigger the allocator.
            del video_latents
            gc.collect()
            torch.cuda.empty_cache()

            if i != 0:
                generated_out = generated[cur_overlap:]
                mask_out = mask_i[cur_overlap:]
                warp_out = warp_i[cur_overlap:]
            else:
                generated_out = generated
                mask_out = mask_i
                warp_out = warp_i

            # Strip the replicate-padding added in read_chunk back off - the
            # model ran on (padded_height, padded_width), but the file on
            # disk should only ever contain the real (height, width) frame.
            if pad_bottom or pad_right:
                generated_out = generated_out[:, :, :height, :width]
                mask_out = mask_out[:, :, :height, :width]
                warp_out = warp_out[:, :, :height, :width]

            # The model is only conditioned on `mask` to tell it where the
            # occlusion holes are - nothing was previously stopping its output
            # from being used verbatim outside those holes too, where warp_i
            # already had a correct reprojected pixel. SVD's non-causal
            # temporal attention means frames near the trailing edge of a
            # diffusion window can drift/hallucinate well outside the masked
            # region (e.g. a fast-motion effect a couple frames before it
            # actually starts in the source). Compositing back through the
            # mask here means the model can only actually change pixels
            # inside the holes it was asked to fill.
            generated_out = mask_out * generated_out + (1 - mask_out) * warp_out

            # OPTIMIZATION: write this chunk to disk now instead of appending to
            # a list that holds the entire output video in RAM until the end.
            # Each chunk gets its own segment file, checkpointed immediately
            # after it's cleanly written - see FFmpegSegmentWriter/main()'s
            # docstring for why (resumable, no moov-atom finalization risk).
            gen_u8 = _to_uint8_rgb(generated_out)
            seg_name = f"seg_{len(segments):06d}.mkv"
            seg_writer = FFmpegSegmentWriter(os.path.join(checkpoint_dir, seg_name), fps, width, height)
            seg_writer.write(gen_u8)
            seg_writer.release()
            segments.append(seg_name)

            del mask_i, warp_i, generated_out, gen_u8

            i += frames_chunk - overlap
            next_i = i
            torch.save(generated, tail_path)
            save_checkpoint()
            progress.finish_iter()

            if max_iters is not None and iter_index >= max_iters:
                print(f"==> Stopping early: reached max_iters={max_iters}")
                break

    if not completed:
        print(
            f"\n==> Stopped early - {len(segments)} chunk(s) checkpointed at {checkpoint_dir}. "
            f"Rerun the same command (resume=True, the default) to continue."
        )
        return

    right_eye_path = os.path.join(save_dir, f"{video_name}_right.mkv")
    segment_paths = [os.path.join(checkpoint_dir, s) for s in segments]
    _concat_segments(segment_paths, right_eye_path, fps, width, height)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)

    print(f"\n==> Right-eye video written to: {right_eye_path}")
    print(f"==> Resolution: {width}x{height} (crop the original video the same way before combining)")
    src = meta.get("source_video_path")
    if src:
        sbs_out = os.path.join(save_dir, f"{video_name}_sbs.mkv")
        anaglyph_out = os.path.join(save_dir, f"{video_name}_anaglyph.mp4")
        # RIGHT always goes through `setpts=N/({fps}*TB)`, never `fps={fps}`.
        # Matroska's muxer can't represent a timescale finer than 1ms
        # (verified empirically, -video_track_timescale/-enc_time_base are
        # no-ops for it), and for 29.97fps (period 1001/30000s = 33.3667ms)
        # that 1ms grid always rounds down to 33ms - not harmless +-0.5ms
        # jitter, but a consistent ~0.367ms/frame bias that compounds across
        # every chunk FFmpegSegmentWriter/_concat_segments writes and joins.
        # Confirmed on an actual 604-frame/301-chunk run: right.mkv's own
        # frame PTS drifted to -220ms (6.6 frames) by the last frame, despite
        # the decoded frame count matching the source exactly (a timestamp
        # bug, not a dropped/duplicated-content bug). Since chunking already
        # guarantees right.mkv has exactly one frame per source frame in
        # order (verified via -count_frames AND by extracting and visually
        # diffing frames straddling an actual scene cut), its presentation
        # timing can be rebuilt from scratch by frame index instead (N =
        # 0-based frame counter, TB = this stream's own timebase) - the
        # drifted input PTS setpts reads are irrelevant to what it writes,
        # so the accumulated bias can't carry through.
        #
        # LEFT uses the *same* setpts treatment whenever stride == 1 (no
        # real up/downsampling needed - verified frame-for-frame parity with
        # right, see above) - NOT `fps={fps}`, even though the source file's
        # own timebase doesn't suffer the mkv-quantization problem above.
        # Mixing a PTS-driven filter (fps=, which snaps to nearest output
        # tick and can therefore select frame N on one side vs N+-1 on the
        # other right at a boundary) on one side with an index-driven filter
        # (setpts) on the other reintroduced a 1-frame skew exactly at hard
        # cuts even after fixing right's own drift above - confirmed by
        # pulling a frame straddling a real cut from the combined output and
        # seeing left still on the old scene while right had already cut.
        # Two filters with different selection logic simply don't agree on
        # exactly which input frame lands in which output slot near a
        # boundary, even when both are fed frame-accurate input. Making both
        # sides index-driven removes the ambiguity instead of trying to keep
        # both filters' rounding in sync.
        #
        # stride > 1 (target_fps < source fps) is the one case that still
        # needs `fps={fps}` on LEFT: right.mkv only has one frame per
        # *sampled* source frame, so left genuinely needs real frames
        # dropped, not just relabeled - see the NOTE below.
        stride = meta.get("stride", 1)
        left_time_filter = f"fps={fps}" if stride != 1 else f"setpts=N/({fps}*TB)"
        # setsar=2/1 doubles the reported DAR (e.g. 32:9 -> 64:9 for a
        # 3840x1080 frame) via the H.264 VUI's sample aspect ratio - a
        # secondary hint some players use. The signal that actually matters
        # to Kodi is Matroska's native StereoMode element, written via
        # -metadata:s:v:0 stereo_mode=left_right - confirmed by diffing
        # ffprobe output against a commercial 3D-SBS mkv: without this tag
        # (e.g. plain mp4, which has no reliably-supported equivalent),
        # Kodi doesn't auto-switch to 3D. Needs an .mkv container - mp4 has
        # no standard field for it. Pixel data is untouched either way,
        # only container/codec-level metadata.
        print("\n==> To combine into side-by-side 3D with ffmpeg (Kodi-compatible):")
        print(
            f'    ffmpeg -i "{src}" -i "{right_eye_path}" -filter_complex '
            f'"[0:v]{left_time_filter},crop={width}:{height}:0:0[left];'
            f'[1:v]setpts=N/({fps}*TB)[right];[left][right]hstack,setsar=2/1" '
            f'-c:v libx264 -pix_fmt yuv420p -crf 18 '
            f'-metadata:s:v:0 stereo_mode=left_right "{sbs_out}"'
        )
        print("\n==> Or into red/cyan anaglyph (ffmpeg has a built-in filter for this):")
        print(
            f'    ffmpeg -i "{src}" -i "{right_eye_path}" -filter_complex '
            f'"[0:v]{left_time_filter},crop={width}:{height}:0:0[left];'
            f'[1:v]setpts=N/({fps}*TB)[right];[left][right]anaglyph=rc" '
            f'-c:v libx264 -crf 18 "{anaglyph_out}"'
        )
        if stride != 1:
            print(
                f"\n    NOTE: this run used stride={stride} (target_fps < source fps), "
                f"so the original video runs at a different frame rate than the right-eye "
                f"output - the command above already accounts for this (LEFT uses `fps={fps}` "
                f"to actually drop frames down to the sampled rate; RIGHT uses setpts since it "
                f"has no extra frames to drop)."
            )


if __name__ == "__main__":
    Fire(main)
