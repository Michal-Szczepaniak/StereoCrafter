import gc
import json
import os
import shutil
import sys
import time
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.training_utils import set_seed
from fire import Fire
from decord import VideoReader, cpu

from dependency.DepthCrafter.depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from dependency.DepthCrafter.depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from dependency.DepthCrafter.depthcrafter.utils import vis_sequence_depth

from Forward_Warp import forward_warp
from splat_store import create_store, open_store, write_meta, ffv1_encode, ffv1_decode

DEPTH_QUANT_LEVELS = 65535  # uint16 - quantization step is (chunk range)/65535,
                             # e.g. a factor of ~6500x finer than a 0.0001 error
                             # tolerance on a depth range of ~1.0


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


class LiveProgress:
    """One continuously-updating status line covering both levels of
    progress this pass has (outer depth-chunk/ETA, and the per-batch warp
    call within it) instead of two separate noisy print streams - the
    outer "Splatting chunk i/N" print and one line per batch. Redraws in
    place via \\r; only commits a real newline once per outer chunk
    (finish_chunk), so nothing scrolls except one line per chunk.

    1:1 port of inpainting_inference.py's LiveProgress (same header/ETA
    mechanics) - the only difference is the sub-level: splatting has no
    per-tile denoising steps, so the sub-level here is the per-batch warp
    call instead of tile/step."""

    def __init__(self, num_frames: int, total_chunks: int):
        self.num_frames = num_frames
        self.total_chunks = total_chunks
        self._header = ""
        self._batch_idx = 0
        self._batch_total = 0
        self._last_len = 0

    def set_header(self, frame_offset, chunk_index, chunk_frames, timing_str):
        self._header = (
            f"Splatting: {frame_offset}/{self.num_frames} "
            f"(chunk {chunk_index}/{self.total_chunks}, {chunk_frames} frames){timing_str}"
        )
        self._batch_idx = self._batch_total = 0
        self._render()

    def set_batch(self, batch_idx: int, batch_total: int):
        self._batch_idx, self._batch_total = batch_idx, batch_total
        self._render()

    def _render(self):
        line = self._header
        if self._batch_total:
            line += f" | batch {self._batch_idx}/{self._batch_total}"
        pad = max(self._last_len - len(line), 0)
        sys.stdout.write("\r" + line + " " * pad)
        sys.stdout.flush()
        self._last_len = len(line)

    def finish_chunk(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


def _store_is_complete(store_dir, expected_params):
    """True if store_dir already holds a finished warp/mask store matching
    expected_params exactly - i.e. this exact stage-1 invocation has nothing
    left to do.

    Checked BEFORE constructing DepthCrafterDemo (model load) or calling
    infer() at all, specifically so that resuming a pipeline where stage 1
    already finished (but stage 2 got interrupted) skips straight past
    stage 1 instead of redoing the whole multi-hour depth pass from frame 0 -
    infer()'s own checkpoint only covers a mid-depth-pass crash, and is
    deleted the moment stage 1 finishes successfully (see main()), so a
    completed run leaves nothing for infer() itself to resume from.

    Requires the depth checkpoint dir to be ABSENT: main() only removes it
    after splatting (which has no resume of its own) also completes, so its
    presence means the prior run was interrupted, not finished - falling
    through to the normal (slower, but safe) checkpoint/resume path.
    """
    checkpoint_dir = os.path.join(store_dir, ".depth_checkpoint")
    if os.path.exists(checkpoint_dir):
        return False
    try:
        warp, mask, meta = open_store(store_dir, mode="r")
    except (FileNotFoundError, ValueError):
        return False
    if meta.get("params") != expected_params:
        return False
    num_frames, height, width = meta.get("num_frames"), meta.get("height"), meta.get("width")
    return (
        warp.shape == (num_frames, height, width, 3)
        and mask.shape == (num_frames, height, width)
    )


def get_video_info(video_path, max_res, target_fps=-1, dataset="open"):
    vid_probe = VideoReader(video_path, ctx=cpu(0))

    original_height, original_width = vid_probe.get_batch([0]).shape[1:3]
    original_num_frames = len(vid_probe)
    source_fps = vid_probe.get_avg_fps()

    if dataset == "open":
        height = round(original_height / 64) * 64
        width = round(original_width / 64) * 64

        if max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale / 64) * 64
            width = round(original_width * scale / 64) * 64
    else:
        raise ValueError("Only dataset='open' is currently supported.")

    fps = source_fps if target_fps == -1 else target_fps
    stride = max(round(source_fps / fps), 1)

    num_sampled_frames = (original_num_frames + stride - 1) // stride

    print("==> original video shape:", (original_num_frames, original_height, original_width, 3))
    print("==> processing resolution:", (height, width))
    print("==> source FPS:", source_fps)
    print("==> target FPS:", fps)
    print("==> frame stride:", stride)
    print("==> sampled frames:", num_sampled_frames)

    return (
        original_num_frames,
        original_height,
        original_width,
        source_fps,
        fps,
        stride,
        height,
        width,
        num_sampled_frames,
    )


def read_video_chunk_streaming(vid, start, end, stride):
    indices = list(range(start * stride, end * stride, stride))
    frames = vid.get_batch(indices).asnumpy().astype(np.float32) / 255.0
    return frames


class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_trained_path: str,
        # None (all model components resident on GPU) is faster than
        # "model" (moves components CPU<->GPU as needed) - measured ~40%
        # faster per denoising step - but uses ~2.5-3GB more peak VRAM
        # (~8.6GB -> ~11-12GB on a 12GB card, empirically noisy run-to-run).
        # Safe as a default now that infer() checkpoints every chunk to disk
        # (see checkpoint_dir below): an OOM kill only loses the in-flight
        # chunk, a rerun of the same command resumes from the last completed
        # one instead of redoing the whole run.
        cpu_offload: Optional[str] = None,
        attention_slicing: bool = True,
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_trained_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        if cpu_offload is not None:
            if cpu_offload == "sequential":
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to("cuda")
        if attention_slicing:
            self.pipe.enable_attention_slicing()

    def infer(
        self,
        input_video_path: str,
        output_dir: str,
        process_length: int = -1,
        num_denoising_steps: int = 8,
        # 1.0 = classifier-free guidance off, matching DepthCrafter's own
        # upstream default. Anything > 1 makes the pipeline run a second full
        # UNet forward pass per denoising step (see do_classifier_free_guidance
        # in depth_crafter_ppl.py) - roughly doubles wall time for a change
        # that measured within the normal frame-to-frame noise floor on an
        # A/B test (mean hole-fraction diff ~1.1pp vs ~0.9pp baseline noise).
        guidance_scale: float = 1.0,
        window_size: int = 70,
        window_overlap: int = 25,
        stream_overlap: int = 25,
        max_res: int = 1024,
        dataset: str = "open",
        target_fps: int = -1,
        seed: int = 42,
        track_time: bool = False,
        save_depth: bool = False,
        chunk_size: int = 110,
        resume: bool = True,
    ):
        """window_size/window_overlap control DepthCrafter's OWN internal
        sliding-window inference within a single self.pipe() call - these
        must satisfy window_overlap < window_size, that's a constraint of
        the model's own windowing loop, not something we chose.

        Our own outer streaming loop (chunk_size) re-reads `window_overlap`
        frames of leading context at each chunk boundary and feeds the
        previous chunk's final `window_overlap` frames of *latents*
        (self.pipe.last_tail_latents) into the next chunk's own
        `carry_latents`. This makes an outer chunk boundary crossfade in
        latent space exactly the way DepthCrafter blends its own internal
        windows - no separate scale/shift correction on the decoded depth
        needed, and no drift/step-jump possible at the seam since it's the
        same mechanism the model already uses internally, just carried
        across separate self.pipe() calls instead of within one.

        `stream_overlap`/--chunk_overlap is accepted for CLI compatibility
        but ignored: the re-feed/discard amount is now always
        window_overlap, since that's what carry_latents' shape has to match.

        chunk_size must be greater than window_size, not just window_overlap
        - a chunk whose own frame count doesn't exceed window_size never
        populates more than one internal window, so it can't produce a tail
        to hand off to the next chunk.

        resume: if a checkpoint from a previous call is found under
        `{output_dir}/.depth_checkpoint` and its recorded parameters match
        this call's, continue from the last completed chunk instead of
        starting over - each chunk's depth array and latent carry-over state
        are written to disk as soon as they're produced, so a crash (an OOM
        under cpu_offload=None, a killed process, etc.) only costs the
        in-flight chunk. Set resume=False to force a clean restart (wipes
        any existing checkpoint for this output_dir first). The checkpoint
        is deleted automatically once splatting also completes successfully
        (see main()).
        """
        set_seed(seed)

        if window_overlap >= window_size:
            raise ValueError(
                f"window_overlap ({window_overlap}) must be less than window_size "
                f"({window_size}) - this is DepthCrafter's own internal windowing "
                f"constraint, unrelated to chunk_size/stream_overlap."
            )

        if stream_overlap != window_overlap:
            print(
                f"==> NOTE: stream_overlap ({stream_overlap}) is ignored - "
                f"cross-chunk continuity now uses window_overlap ({window_overlap}) "
                f"for both the leading-context re-feed and the latent carry-over."
            )

        (
            original_num_frames,
            original_height,
            original_width,
            source_fps,
            target_fps,
            stride,
            processing_height,
            processing_width,
            total_frames,
        ) = get_video_info(
            input_video_path,
            max_res,
            target_fps,
            dataset,
        )

        if process_length != -1:
            total_frames = min(process_length, total_frames)

        print("==> total frames to process:", total_frames)
        print("==> streaming chunk size:", chunk_size)
        print(f"==> DepthCrafter internal window: size={window_size}, overlap={window_overlap}")
        print("==> frame stride (video -> depth):", stride)

        if chunk_size <= window_size:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be greater than window_size "
                f"({window_size}) - otherwise a chunk never populates more than one "
                f"internal window and can't hand off a latent tail to the next chunk."
            )

        vid = VideoReader(
            input_video_path,
            ctx=cpu(0),
            width=processing_width,
            height=processing_height,
        )

        # ------------------------------------------------------------------
        # PASS 1: run DepthCrafter chunk-by-chunk, save depth chunks to disk.
        # Overlap is retained for temporal context; only new frames from each
        # chunk are kept in the saved result. Persisted under output_dir
        # (not a tempdir - see resume above) so a crash can pick back up
        # instead of losing the whole depth pass.
        # ------------------------------------------------------------------

        checkpoint_dir = os.path.join(output_dir, ".depth_checkpoint")
        manifest_path = os.path.join(checkpoint_dir, "manifest.json")
        latents_path = os.path.join(checkpoint_dir, "carry_latents.pt")

        run_params = {
            "input_video_path": os.path.abspath(input_video_path),
            "chunk_size": chunk_size,
            "window_size": window_size,
            "window_overlap": window_overlap,
            "max_res": max_res,
            "total_frames": total_frames,
            "guidance_scale": guidance_scale,
            "num_denoising_steps": num_denoising_steps,
            "seed": seed,
            "target_fps": target_fps,
        }

        output_start = 0
        chunk_files = []
        chunk_meta = []  # [{"min","max","frames"}, ...] parallel to chunk_files -
                          # per-CHUNK quantization range (not the global one,
                          # which isn't known until the whole pass finishes) -
                          # see DEPTH_QUANT_LEVELS / DepthSplatting for the read side
        global_min = np.inf
        global_max = -np.inf
        prev_tail_latents = None  # last `window_overlap` frames of *latents*
                                   # from the previous chunk (self.pipe.last_tail_latents)

        if not resume:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        elif os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            if (
                manifest.get("params") == run_params
                and manifest.get("chunk_files")
                and manifest.get("chunk_meta") is not None  # absent = pre-quantization-era checkpoint, incompatible
            ):
                output_start = manifest["output_start"]
                chunk_files = [os.path.join(checkpoint_dir, name) for name in manifest["chunk_files"]]
                chunk_meta = manifest["chunk_meta"]
                global_min = manifest["global_min"]
                global_max = manifest["global_max"]
                if output_start > 0 and os.path.exists(latents_path):
                    prev_tail_latents = torch.load(latents_path, map_location="cuda")
                print(
                    f"==> RESUMING depth pass from checkpoint: {len(chunk_files)} "
                    f"chunk(s) already done, continuing at output frame {output_start}"
                )
            else:
                print("==> checkpoint found but parameters differ from this run - starting fresh")
                shutil.rmtree(checkpoint_dir, ignore_errors=True)

        os.makedirs(checkpoint_dir, exist_ok=True)
        print("==> depth checkpoint directory:", checkpoint_dir)

        def save_checkpoint():
            tmp_path = manifest_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(
                    {
                        "params": run_params,
                        "output_start": output_start,
                        "chunk_files": [os.path.basename(p) for p in chunk_files],
                        "chunk_meta": chunk_meta,
                        "global_min": float(global_min),
                        "global_max": float(global_max),
                    },
                    f,
                )
            os.replace(tmp_path, manifest_path)

        try:
            while output_start < total_frames:
                if output_start == 0:
                    input_start = 0
                else:
                    input_start = max(0, output_start - window_overlap)

                input_end = min(output_start + chunk_size, total_frames)

                print(
                    f"\n==> DepthCrafter chunk: input {input_start}:{input_end}, "
                    f"output starting at {output_start}"
                )

                frames = read_video_chunk_streaming(vid, input_start, input_end, stride)

                print(
                    "    chunk shape:", frames.shape,
                    "RAM input:", f"{frames.nbytes / 1024**2:.1f} MiB",
                )

                with torch.inference_mode():
                    result = self.pipe(
                        frames,
                        height=frames.shape[1],
                        width=frames.shape[2],
                        output_type="np",
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_denoising_steps,
                        window_size=window_size,
                        overlap=window_overlap,
                        track_time=track_time,
                        carry_latents=prev_tail_latents,
                    ).frames[0]

                # Stash this call's tail latents (already crossfaded/continuous by
                # construction) for the *next* chunk's carry_latents, before doing
                # any further per-chunk processing below.
                prev_tail_latents = self.pipe.last_tail_latents

                result = result.sum(-1) / result.shape[-1]

                print(
                    "    raw depth:", result.shape,
                    "finite=", np.isfinite(result).all(),
                    "min=", np.nanmin(result),
                    "max=", np.nanmax(result),
                )

                tensor_res = torch.from_numpy(result).unsqueeze(1).float().cuda()
                result = F.interpolate(
                    tensor_res,
                    size=(original_height, original_width),
                    mode="bilinear",
                    align_corners=False,
                )
                result = result[:, 0].cpu().numpy()

                finite = np.isfinite(result)
                if not finite.all():
                    raise RuntimeError("DepthCrafter produced NaN/Inf depth in a streaming chunk.")

                # discard = how many frames at the start of this chunk's output
                # duplicate the tail already emitted by the previous chunk (now
                # continuous with it via the latent carry-over, so this is a
                # straight trim - no scale/shift correction needed). 0 for the
                # first chunk.
                discard = output_start - input_start if output_start > 0 else 0

                if discard > 0:
                    result = result[discard:]

                remaining = total_frames - output_start
                result = result[:remaining]

                chunk_min = result.min()
                chunk_max = result.max()
                global_min = min(global_min, chunk_min)
                global_max = max(global_max, chunk_max)

                # Quantized to uint16 using THIS CHUNK's own min/max (not the
                # global one - that isn't known until the whole pass finishes,
                # since chunks are written as they're produced). Dequantized
                # back to float32 using this same range in DepthSplatting,
                # THEN the global normalization is applied exactly as before -
                # so quantization only adds rounding error, never a range
                # mismatch. Error is ~(chunk_max-chunk_min)/65535 - orders of
                # magnitude below a 0.0001 tolerance on a ~1.0-wide depth
                # range, and disparity (the only thing depth is ever used
                # for) is quantized to whole/half pixels anyway.
                chunk_path = os.path.join(checkpoint_dir, f"depth_{len(chunk_files):06d}.mkv")
                chunk_range = max(float(chunk_max) - float(chunk_min), 1e-12)
                quantized = np.round((result - chunk_min) / chunk_range * DEPTH_QUANT_LEVELS).astype(np.uint16)
                ffv1_encode(quantized, chunk_path, "gray16le", original_width, original_height)
                chunk_files.append(chunk_path)
                chunk_meta.append({
                    "min": float(chunk_min), "max": float(chunk_max), "frames": int(result.shape[0]),
                })
                if prev_tail_latents is not None:
                    torch.save(prev_tail_latents, latents_path)

                print(f"    saved {result.shape[0]} frames -> {chunk_path}")
                print(f"    chunk depth range: {chunk_min:.6f} .. {chunk_max:.6f}")
                print(f"    global depth range: {global_min:.6f} .. {global_max:.6f}")

                del frames, result, tensor_res
                gc.collect()
                torch.cuda.empty_cache()

                output_start += chunk_size
                save_checkpoint()

            print("\n==> Depth pass complete.")
            print(f"==> GLOBAL depth range: {global_min:.6f} .. {global_max:.6f}")

            if not np.isfinite(global_min) or not np.isfinite(global_max):
                raise RuntimeError("Global depth range is invalid.")
            if global_max <= global_min:
                raise RuntimeError(f"Depth has no usable range: min={global_min}, max={global_max}")

            print("\n==> Depth chunks are ready for splatting.")

            return (
                chunk_files,
                chunk_meta,
                global_min,
                global_max,
                target_fps,
                stride,  # OPTIMIZATION/FIX: DepthSplatting needs this to read the
                         # *same* video frames that were used to build each depth chunk.
                original_height,
                original_width,
                checkpoint_dir,
            )

        except Exception:
            print(
                f"==> Depth pass failed - checkpoint kept at {checkpoint_dir} "
                f"({len(chunk_files)} chunk(s) done). Rerun the same command to resume."
            )
            raise


class ForwardWarpStereo(nn.Module):
    def __init__(self, eps=1e-6, occlu_map=False):
        super(ForwardWarpStereo, self).__init__()
        self.eps = eps
        self.occlu_map = occlu_map
        self.fw = forward_warp()

    def forward(self, im, disp):
        im = im.contiguous()
        disp = disp.contiguous()
        weights_map = disp - disp.min()
        weights_map = (1.414) ** weights_map
        flow = -disp.squeeze(1)
        dummy_flow = torch.zeros_like(flow, requires_grad=False)
        flow = torch.stack((flow, dummy_flow), dim=-1)
        res_accum = self.fw(im * weights_map, flow)
        mask = self.fw(weights_map, flow)
        mask.clamp_(min=self.eps)
        res = res_accum / mask
        if not self.occlu_map:
            return res
        else:
            ones = torch.ones_like(disp, requires_grad=False)
            occlu_map = self.fw(ones, flow)
            occlu_map.clamp_(0.0, 1.0)
            occlu_map = 1.0 - occlu_map
            return res, occlu_map


def DepthSplatting(
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
    store_params=None,
    compress_store=True,
):
    """Stream saved DepthCrafter depth chunks through the depth-splatting
    stage and write ONLY what the inpainting stage reads - the warped
    (right-eye) image and the occlusion mask - under `store_dir` (see
    splat_store.py). "left" is not duplicated here at all; the inpaint
    stage reads it straight from `input_video_path`.

    Each depth chunk is itself FFV1-compressed + uint16-quantized (see
    DEPTH_QUANT_LEVELS and the write side in DepthCrafterDemo.infer) - decoded
    and dequantized here using that chunk's own (min, max) from chunk_meta
    BEFORE the global normalization below, so quantization only adds a tiny
    per-chunk rounding error, it never depends on knowing the eventual
    episode-wide range at write time.

    compress_store (default True): FFV1-compressed WARP/MASK store instead
    of raw .npy - measured ~7x smaller on real content with a verified
    bit-exact round-trip, at a CPU decode cost stage 2 never notices (see
    splat_store.py's module docstring for the numbers and why plain H.264
    was rejected). Set False to fall back to the original uncompressed
    format if you ever need to debug/compare against it directly. (This
    only affects warp/mask - the depth checkpoint read below is always
    quantized+compressed, independent of this flag.)

    FIX (carried over): depth chunk index i was computed from source video
    frame i*stride (DepthCrafter runs on a temporally-subsampled clip
    whenever stride > 1, i.e. whenever target_fps < source fps). This reads
    the same strided frame indices back out of the source video, so image
    and depth stay paired.
    """
    vid_reader = VideoReader(input_video_path, ctx=cpu(0))
    native_num_frames = len(vid_reader)

    total_depth_frames = sum(m["frames"] for m in chunk_meta)
    # Number of (depth, frame) pairs we can actually produce, bounded by how
    # many strided source frames exist.
    max_pairs_from_source = (native_num_frames + stride - 1) // stride
    num_frames = min(total_depth_frames, max_pairs_from_source)

    print(f"==> splatting {num_frames} frames (stride={stride})")
    print(f"==> depth chunks: {len(chunk_files)}")
    print(f"==> depth range: {global_min:.6f} .. {global_max:.6f}")

    first_frame = vid_reader[0].asnumpy()
    height, width = first_frame.shape[:2]
    del first_frame

    if (height, width) != (original_height, original_width):
        print(
            f"==> WARNING: native frame size {(height, width)} does not match "
            f"the resolution depth was computed/resized at {(original_height, original_width)}"
        )

    warp_store, mask_store = create_store(store_dir, num_frames, height, width, compress=compress_store)
    write_meta(
        store_dir,
        fps=target_fps,
        stride=stride,
        num_frames=num_frames,
        height=height,
        width=width,
        source_video_path=os.path.abspath(input_video_path),
        params=store_params,
    )
    print(f"==> writing splat store to: {store_dir}")
    print(f"==> ({num_frames} x {height} x {width}) warp uint8 + mask uint8, no grid, no depth-vis")

    stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()

    frame_offset = 0

    # OPTIMIZATION: rolling average of per-chunk wall time, just to print an
    # ETA - the last chunk's frame count can differ (see chunk_frames
    # clamping below), so this is an estimate, not an exact countdown.
    chunk_durations = []
    chunk_start = None
    progress = LiveProgress(num_frames, len(chunk_files))

    for chunk_index, chunk_path in enumerate(chunk_files):
        meta_c = chunk_meta[chunk_index]
        quantized = ffv1_decode(chunk_path, "gray16le", width, height, channels=None, dtype=np.uint16)
        depth_chunk = (
            meta_c["min"]
            + quantized.astype(np.float32) / DEPTH_QUANT_LEVELS * (meta_c["max"] - meta_c["min"])
        )
        chunk_frames = min(depth_chunk.shape[0], num_frames - frame_offset)
        if chunk_frames <= 0:
            break

        if chunk_start is not None:
            chunk_durations.append(time.monotonic() - chunk_start)
        chunk_start = time.monotonic()

        if chunk_durations:
            avg_chunk_time = sum(chunk_durations) / len(chunk_durations)
            last_chunk_time = chunk_durations[-1]
            remaining_chunks = max(len(chunk_files) - chunk_index, 0)
            eta = _format_duration(avg_chunk_time * remaining_chunks)
            timing_str = (
                f", last_chunk {last_chunk_time:.1f}s, avg_chunk {avg_chunk_time:.1f}s, "
                f"ETA {eta}"
            )
        else:
            timing_str = ""
        progress.set_header(frame_offset, chunk_index + 1, chunk_frames, timing_str)

        total_batches = (chunk_frames + batch_size - 1) // batch_size
        for i in range(0, chunk_frames, batch_size):
            progress.set_batch(i // batch_size + 1, total_batches)
            end = min(i + batch_size, chunk_frames)

            sample_start = frame_offset + i
            sample_end = frame_offset + end

            # Sampled (depth-space) indices -> native video frame indices.
            native_indices = list(
                range(
                    sample_start * stride,
                    min(sample_end * stride, native_num_frames),
                    stride,
                )
            )
            if not native_indices:
                break

            batch_frames = (
                vid_reader.get_batch(native_indices).asnumpy().astype(np.float32) / 255.0
            )
            n = len(native_indices)  # keep depth aligned if clipped near EOF
            batch_depth = np.asarray(depth_chunk[i : i + n], dtype=np.float32)

            batch_depth = (batch_depth - global_min) / (global_max - global_min)
            batch_depth = np.clip(batch_depth, 0.0, 1.0)

            left_video = torch.from_numpy(batch_frames).permute(0, 3, 1, 2).float().cuda()
            disp_map = torch.from_numpy(batch_depth).unsqueeze(1).float().cuda()
            disp_map = disp_map * 2.0 - 1.0
            disp_map = disp_map * max_disp

            with torch.no_grad():
                right_video, occlusion_mask = stereo_projector(left_video, disp_map)

            # OPTIMIZATION: write only the two arrays the inpaint stage
            # reads - no grid assembly, no depth-vis channel, no 3x
            # replication of a single-channel mask, no color-space convert.
            right_u8 = (
                right_video.clamp(0, 1).mul(255).to(torch.uint8)
                .permute(0, 2, 3, 1).contiguous().cpu().numpy()
            )
            mask_u8 = (
                occlusion_mask.clamp(0, 1).mul(255).to(torch.uint8)
                .squeeze(1).contiguous().cpu().numpy()
            )

            warp_store[sample_start : sample_start + n] = right_u8
            mask_store[sample_start : sample_start + n] = mask_u8

            del (
                batch_frames, batch_depth, left_video, disp_map,
                right_video, occlusion_mask, right_u8, mask_u8,
            )

        # Flush this chunk's writes to disk and release GPU memory before
        # starting the next depth chunk - same cadence as before, just no
        # video writer to release.
        warp_store.flush()
        mask_store.flush()
        gc.collect()
        torch.cuda.empty_cache()

        frame_offset += chunk_frames
        del depth_chunk

        # Free this chunk's depth file now that it's safely flushed into
        # warp_store/mask_store - depth (float32, 1ch) and warp+mask
        # (uint8, 3ch+1ch) are ~the same bytes/pixel, so without this,
        # PASS 2 needs a second full copy of the store's total size on top
        # of the still-undeleted depth checkpoint (main() only removes
        # checkpoint_dir after this whole function returns) - on a
        # multi-hundred-GB episode that's an ENOSPC guarantee, not a risk.
        # Tradeoff: a PASS-2 failure for a *non*-disk reason after this
        # point can no longer redo splatting from chunk 0 (this chunk's
        # depth file is gone) - accepted since the alternative (no delete)
        # fails every time anyway.
        os.remove(chunk_path)
        progress.finish_chunk()

    warp_store.flush()
    mask_store.flush()
    # Compressed writer: final flush + write the group index (no-op for the
    # raw memmap format, which has no .close()).
    if hasattr(warp_store, "close"):
        warp_store.close()
    if hasattr(mask_store, "close"):
        mask_store.close()

    print("==> Depth splatting complete.")
    print(f"==> wrote: {store_dir}")
    return store_dir


def main(
    input_video_path: str,
    output_dir: str,
    unet_path: str,
    pre_trained_path: str,
    max_disp: float = 20.0,
    process_length: int = -1,
    batch_size: int = 10,
    max_res: int = 768,
    chunk_size: int = 110,
    chunk_overlap: int = 25,
    window_size: int = 70,
    window_overlap: int = 25,
    cpu_offload: Optional[str] = None,
    attention_slicing: bool = True,
    target_fps: int = -1,
    num_denoising_steps: int = 8,
    guidance_scale: float = 1.0,
    resume: bool = True,
    compress_store: bool = True,
):
    """NOTE: --output_video_path is now --output_dir - this stage no longer
    writes an mp4, it writes a directory (splat_store.py's format -
    FFV1-compressed by default, see compress_store) - point
    inpainting_inference.py's --splat_store_dir at this directory.

    chunk_overlap is now ignored (kept only so old scripts/env vars don't
    break) - cross-chunk continuity uses window_overlap for both the
    leading-context re-feed and a latent-space carry-over between chunks
    (see DepthCrafterDemo.infer's docstring), which crossfades an outer
    chunk boundary exactly like DepthCrafter's own internal window
    transitions. window_overlap MUST be less than window_size - that's the
    model's own internal windowing constraint.

    resume: continue a previous run's depth pass from its checkpoint under
    {output_dir}/.depth_checkpoint if the parameters match (see
    DepthCrafterDemo.infer). The checkpoint is only deleted once splatting
    also completes successfully - if splatting fails, rerunning this same
    command skips straight back to splatting instead of redoing the depth
    pass. Set resume=False to force a clean restart.

    Also: if output_dir already holds a *complete* store for these exact
    params (stage 1 finished on a previous run, e.g. while stage 2 got
    interrupted and the whole pipeline was rerun from the top), this returns
    immediately without loading the model or touching the GPU at all -
    infer()'s own checkpoint can't cover this case since it's deleted the
    moment stage 1 finishes (see _store_is_complete's docstring).
    """
    store_params = {
        "input_video_path": os.path.abspath(input_video_path),
        "max_disp": max_disp,
        "process_length": process_length,
        "max_res": max_res,
        "chunk_size": chunk_size,
        "window_size": window_size,
        "window_overlap": window_overlap,
        "target_fps": target_fps,
        "num_denoising_steps": num_denoising_steps,
        "guidance_scale": guidance_scale,
    }
    if resume and _store_is_complete(output_dir, store_params):
        print(f"==> Stage 1 already complete for these params - reusing existing store at {output_dir}")
        return

    depthcrafter_demo = DepthCrafterDemo(
        unet_path=unet_path,
        pre_trained_path=pre_trained_path,
        cpu_offload=cpu_offload,
        attention_slicing=attention_slicing,
    )

    (
        chunk_files,
        chunk_meta,
        global_min,
        global_max,
        target_fps,
        stride,
        original_height,
        original_width,
        checkpoint_dir,
    ) = depthcrafter_demo.infer(
        input_video_path,
        output_dir,
        process_length=process_length,
        max_res=max_res,
        chunk_size=chunk_size,
        stream_overlap=chunk_overlap,
        window_size=window_size,
        window_overlap=window_overlap,
        target_fps=target_fps,
        num_denoising_steps=num_denoising_steps,
        guidance_scale=guidance_scale,
        resume=resume,
    )

    try:
        DepthSplatting(
            input_video_path,
            output_dir,
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
            store_params=store_params,
            compress_store=compress_store,
        )
    except Exception:
        print(f"==> Splatting failed - depth checkpoint kept at {checkpoint_dir} for resume")
        raise
    else:
        print("==> removing depth checkpoint (run complete)")
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    Fire(main)
