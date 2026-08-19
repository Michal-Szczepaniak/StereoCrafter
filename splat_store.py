"""On-disk interchange format between the depth-splatting stage and the
inpainting stage.

Replaces writing a 2x2 grid mp4 (left | depth-vis on top, occlusion-mask |
warped-right on bottom) that the inpaint stage then had to re-decode. That
format had three problems:

  1. A full quadrant (depth-vis) was written and decoded on the other end
     for nothing - the inpaint stage never reads it.
  2. The "left" frame was duplicated - it's already sitting in the original
     source video, unmodified, one quadrant over.
  3. Everything was round-tripped through an 8-bit YUV420 lossy video codec.
     That's a real quality problem specifically for the occlusion mask: mask
     edges are sharp, non-natural-image content, exactly what chroma
     subsampling + DCT quantization distorts most. The warped image takes a
     second generation-loss hit too, on top of whatever the source video's
     own encoding already cost it.

This module stores only what the inpaint stage reads: the warped (right-eye)
image and the occlusion mask. "left" is not duplicated here - the inpaint
stage reads it directly from the original source video (path recorded in
meta.json).

Two on-disk formats, chosen via create_store(..., compress=...):

RAW (compress=False, the original format): memory-mapped .npy arrays
(numpy.lib.format.open_memmap). Self-describing (shape/dtype in the header),
genuinely random-access (no decoding forward from a keyframe), and each
write/read slice only touches its own pages. The tradeoff is size: at
1920x1080 this is ~4 bytes/pixel uncounted (warp uint8x3 + mask uint8x1) with
zero compression - on a full-length episode (tens of thousands of frames)
that's hundreds of GB.

COMPRESSED (compress=True, default): FFV1 (lossless video codec, via
ffmpeg) in small per-flush groups instead of one big array. MEASURED on a
real 240-frame local splat store (1920x1080, anime source) - see
conversation history for the full breakdown - FFV1 with native RGB pixel
format gave a bit-exact round-trip (verified frame-for-frame identical to
the raw arrays) at roughly 6x smaller for warp and 21x smaller for mask
(mask compresses far better - it's mostly large flat 0/255 regions).
Combined that's roughly a 7x reduction on this content - real ratio depends
on source complexity, but even a conservative fraction of that is a huge
win on a multi-hundred-GB store. Encode/decode is fast (a fraction of a
second per few-hundred-frame group on this machine) - utterly negligible
next to stage 2's per-iteration diffusion cost (seconds to minutes), so
this trades a small amount of otherwise-idle CPU time (the GPU is the
bottleneck throughout this pipeline, see [[project_rocm_no_flash_attention]]
-adjacent reasoning) for a large, real reduction in rental storage cost.

IMPORTANT: plain libx264 (even at -qp 0, even with an explicit RGB pixel
format) was tested and rejected - it round-tripped through an RGB<->YUV444
color-matrix conversion that is NOT bit-exact even without chroma
subsampling, silently reintroducing exactly the precision loss problem
this module was built to get away from (see the mp4-format history above).
FFV1 with a native RGB pixel format (no YUV conversion at all) was the only
codec tested that verified truly lossless. Don't swap in a different codec
here without doing the same bit-exact round-trip check.

Write side (compress=True): frames are buffered in RAM and only encoded to
disk on .flush() - callers already flush once per outer processing chunk
(see depth_splatting_inference.py's DepthSplatting loop) for cadence
reasons unrelated to compression, and that cadence becomes the group
boundary here for free - no separate "group size" concept needed. Writes
must be contiguous (no gaps, no overwrites) - true for every caller in this
codebase (splatting always writes strictly increasing frame ranges).

Read side (compress=True): an index (`<name>_index.json`) records each
group's (start, count, filename). A read of an arbitrary [start:end) slice
decodes only the group(s) it spans (usually one, sometimes two at a
boundary) - a small LRU keeps the last couple of decoded groups around
since inpainting's sliding window with overlap re-reads nearby frames
across consecutive calls.
"""

import json
import os
import subprocess

import numpy as np
from numpy.lib.format import open_memmap

WARP_FILENAME = "warp.npy"
MASK_FILENAME = "mask.npy"
META_FILENAME = "meta.json"

# FFV1 group filenames: "<prefix>_<group_index>.mkv" + "<prefix>_index.json"
WARP_PREFIX = "warp"
MASK_PREFIX = "mask"

_FFV1_CACHE_GROUPS = 2  # decoded groups kept resident - see module docstring


def ffv1_encode(frames, out_path, pix_fmt, width, height):
    """frames: contiguous np.uint8/np.uint16 array (dtype must match
    pix_fmt's bit depth), shape (n, H, W[, C]) matching pix_fmt (rgb24 =
    3ch uint8, gray = 1ch/no trailing dim uint8, gray16le = 1ch uint16 -
    used for quantized depth, see depth_splatting_inference.py).

    Shared by splat_store.py (warp/mask) and depth_splatting_inference.py
    (quantized depth checkpoint) - not splat-store-specific despite living
    in this module, kept here just to have one FFV1 subprocess
    implementation instead of two."""
    frames = np.ascontiguousarray(frames)
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", pix_fmt,
            "-s", f"{width}x{height}", "-r", "24",
            "-i", "-",
            "-c:v", "ffv1", "-level", "3",
            out_path,
        ],
        stdin=subprocess.PIPE,
    )
    proc.stdin.write(frames.tobytes())
    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg FFV1 encode failed (exit {ret}) for {out_path}")


def ffv1_decode(path, pix_fmt, width, height, channels, dtype=np.uint8):
    proc = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error",
            "-i", path,
            "-f", "rawvideo", "-pix_fmt", pix_fmt,
            "-",
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    arr = np.frombuffer(proc.stdout, dtype=dtype)
    shape = (-1, height, width, channels) if channels else (-1, height, width)
    return arr.reshape(shape)


class _CompressedWriter:
    """Buffers contiguous frame writes in RAM, FFV1-encodes to a new group
    file on each .flush() call. See module docstring for why group
    boundaries just follow the caller's own flush cadence."""

    def __init__(self, store_dir, prefix, height, width, channels):
        self._store_dir = store_dir
        self._prefix = prefix
        self._height = height
        self._width = width
        self._channels = channels
        self._pix_fmt = "rgb24" if channels == 3 else "gray"
        self._buffer = []
        self._next_frame = 0
        self._group_index = 0
        self._groups = []  # [{"start", "count", "file"}, ...]

    def __setitem__(self, key, value):
        assert isinstance(key, slice) and key.step in (None, 1), \
            "compressed store only supports contiguous slice writes"
        start, stop = key.start, key.stop
        if start != self._next_frame:
            raise ValueError(
                f"compressed store requires contiguous writes - expected start="
                f"{self._next_frame}, got {start} (no gaps/overwrites/reordering)"
            )
        self._buffer.append(np.array(value, copy=True))
        self._next_frame = stop

    def flush(self):
        if not self._buffer:
            return
        data = self._buffer[0] if len(self._buffer) == 1 else np.concatenate(self._buffer, axis=0)
        n = data.shape[0]
        group_start = self._next_frame - n
        filename = f"{self._prefix}_{self._group_index:06d}.mkv"
        out_path = os.path.join(self._store_dir, filename)
        ffv1_encode(data, out_path, self._pix_fmt, self._width, self._height)
        self._groups.append({"start": group_start, "count": n, "file": filename})
        self._group_index += 1
        self._buffer = []

    def close(self):
        """Final flush + write the group index. Must be called once after
        all writes are done (DepthSplatting's own trailing .flush() calls
        do NOT write the index - call this explicitly, see create_store)."""
        self.flush()
        index_path = os.path.join(self._store_dir, f"{self._prefix}_index.json")
        with open(index_path, "w") as f:
            json.dump(self._groups, f, indent=2)


class _CompressedReader:
    def __init__(self, store_dir, prefix, num_frames, height, width, channels):
        self._store_dir = store_dir
        self._prefix = prefix
        self._height = height
        self._width = width
        self._channels = channels
        self._pix_fmt = "rgb24" if channels == 3 else "gray"
        self.shape = (num_frames, height, width, channels) if channels else (num_frames, height, width)

        index_path = os.path.join(store_dir, f"{prefix}_index.json")
        with open(index_path) as f:
            self._groups = json.load(f)
        self._cache = {}  # filename -> decoded array (small LRU, insertion order)

    def _group_for(self, frame_idx):
        for g in self._groups:
            if g["start"] <= frame_idx < g["start"] + g["count"]:
                return g
        raise IndexError(f"frame {frame_idx} out of range for {self._prefix} store")

    def _decode_group(self, g):
        if g["file"] not in self._cache:
            if len(self._cache) >= _FFV1_CACHE_GROUPS:
                self._cache.pop(next(iter(self._cache)))
            path = os.path.join(self._store_dir, g["file"])
            self._cache[g["file"]] = ffv1_decode(
                path, self._pix_fmt, self._width, self._height, self._channels
            )
        return self._cache[g["file"]]

    def __getitem__(self, key):
        assert isinstance(key, slice) and key.step in (None, 1)
        start, stop = key.start, key.stop
        pieces = []
        i = start
        while i < stop:
            g = self._group_for(i)
            arr = self._decode_group(g)
            local_start = i - g["start"]
            local_stop = min(g["count"], stop - g["start"])
            pieces.append(arr[local_start:local_stop])
            i = g["start"] + local_stop
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)


def create_store(store_dir, num_frames, height, width, compress=True):
    """Preallocate/prepare the on-disk store for writing.

    Returns (warp, mask). Assigning into a contiguous slice (e.g.
    warp[10:20] = ...) writes only that slice.

    compress=True (default): FFV1-backed, see module docstring. Caller MUST
    call warp.close() and mask.close() once after all writes are done (not
    just the per-chunk .flush() calls) to write the group index - see
    DepthSplatting's use of this.
    compress=False: the original raw memmap format - no .close() needed
    (numpy flushes via .flush(), same as before).
    """
    os.makedirs(store_dir, exist_ok=True)
    if compress:
        warp = _CompressedWriter(store_dir, WARP_PREFIX, height, width, channels=3)
        mask = _CompressedWriter(store_dir, MASK_PREFIX, height, width, channels=None)
        return warp, mask

    warp = open_memmap(
        os.path.join(store_dir, WARP_FILENAME),
        mode="w+",
        dtype=np.uint8,
        shape=(num_frames, height, width, 3),
    )
    mask = open_memmap(
        os.path.join(store_dir, MASK_FILENAME),
        mode="w+",
        dtype=np.uint8,
        shape=(num_frames, height, width),
    )
    return warp, mask


def write_meta(store_dir, **meta):
    with open(os.path.join(store_dir, META_FILENAME), "w") as f:
        json.dump(meta, f, indent=2)


def open_store(store_dir, mode="r"):
    """Open an existing store. mode='r' for read-only (inpaint stage).

    Auto-detects RAW vs COMPRESSED format from what's actually on disk
    (warp.npy vs warp_index.json) - callers don't need to know which one a
    given store used.

    Returns (warp, mask, meta_dict). For the raw format shape/dtype come
    from each .npy file's own header; for the compressed format, from
    meta.json's num_frames/height/width (required in that case).
    """
    meta_path = os.path.join(store_dir, META_FILENAME)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    if os.path.exists(os.path.join(store_dir, WARP_FILENAME)):
        warp = open_memmap(os.path.join(store_dir, WARP_FILENAME), mode=mode)
        mask = open_memmap(os.path.join(store_dir, MASK_FILENAME), mode=mode)
        return warp, mask, meta

    index_path = os.path.join(store_dir, f"{WARP_PREFIX}_index.json")
    if not os.path.exists(index_path):
        # Neither format present - most commonly just means store_dir
        # doesn't hold a store yet at all (e.g. a fresh run's
        # _store_is_complete probe). Match open_memmap's behavior above so
        # callers' existing `except (FileNotFoundError, ValueError)`
        # handling covers this case too, instead of hitting a KeyError from
        # an empty/missing meta.json below.
        raise FileNotFoundError(
            f"no splat store found at {store_dir} (neither {WARP_FILENAME} nor "
            f"{os.path.basename(index_path)})"
        )

    num_frames, height, width = meta["num_frames"], meta["height"], meta["width"]
    warp = _CompressedReader(store_dir, WARP_PREFIX, num_frames, height, width, channels=3)
    mask = _CompressedReader(store_dir, MASK_PREFIX, num_frames, height, width, channels=None)
    return warp, mask, meta
