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
image and the occlusion mask, as memory-mapped .npy arrays. "left" is not
duplicated here - the inpaint stage reads it directly from the original
source video (path recorded in meta.json).

Why memmap (numpy.lib.format.open_memmap) instead of a directory of chunk
files or one big in-RAM array:
  - Self-describing: shape/dtype live in the .npy header, so opening for
    read needs no separate spec.
  - Writing a slice touches only that slice's pages - the splatting stage
    never holds more than one batch in RAM; the OS flushes dirty pages to
    disk in the background (plus an explicit .flush() after each chunk).
  - Reading a window touches only that window's pages - the inpaint stage
    never loads the whole array either, and unlike a video codec, access is
    genuinely random: no decoding forward from a keyframe.
"""

import json
import os

import numpy as np
from numpy.lib.format import open_memmap

WARP_FILENAME = "warp.npy"
MASK_FILENAME = "mask.npy"
META_FILENAME = "meta.json"


def create_store(store_dir, num_frames, height, width):
    """Preallocate the on-disk arrays for writing.

    Returns (warp, mask) - both numpy memmaps in 'w+' mode. Assigning into a
    slice (e.g. warp[10:20] = ...) writes only that slice; the rest of the
    array is never materialized in RAM by this process.
    """
    os.makedirs(store_dir, exist_ok=True)
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

    Returns (warp, mask, meta_dict). Shape/dtype come from each .npy file's
    own header, not from meta.json.
    """
    warp = open_memmap(os.path.join(store_dir, WARP_FILENAME), mode=mode)
    mask = open_memmap(os.path.join(store_dir, MASK_FILENAME), mode=mode)

    meta_path = os.path.join(store_dir, META_FILENAME)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    return warp, mask, meta
