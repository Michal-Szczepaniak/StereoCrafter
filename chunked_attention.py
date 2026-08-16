"""Memory-efficient attention fallback for GPUs with no working flash/
efficient SDPA kernel.

Why this exists: on this project's dev GPU (AMD RX 6700 XT, gfx1030/RDNA2),
torch.nn.attention.sdpa_kernel([FLASH_ATTENTION, EFFICIENT_ATTENTION]) raises
"No available kernel" for every shape this pipeline uses - confirmed
empirically, not assumed. ROCm's flash-attention support targets CDNA
(MI-series) and newer RDNA3 GPUs; RDNA2 consumer cards like this one have no
working fused attention kernel in this PyTorch/ROCm build. That forces
diffusers' default AttnProcessor2_0 to fall back to PyTorch's naive "math"
SDPA backend, which materializes the full (seq_len x seq_len) score matrix -
fine for stage 1's small tiles but O(seq_len^2) is catastrophic for larger
inpainting tiles (tile_num=2 needs ~20GB for a single attention call at
~10k tokens/tile, tile_num=1 would need far more - independent of GPU VRAM
size, since it's a quadratic blowup, not a linear one). This is why NVIDIA
users (working flash-attention, O(seq_len) memory) report tile_num=2 fine
on a 24GB card in upstream issues, while this hardware genuinely cannot.

chunked_sdpa below is a standard online-softmax ("flash attention" algorithm)
implementation in pure PyTorch: processes the key/value sequence in blocks
and accumulates a running max/sum instead of materializing the full score
matrix, giving O(seq_len * kv_chunk_size) memory instead of O(seq_len^2) -
at real compute-time cost: measured ~9-11x slower than the equivalent
tile_num=4 (small-tile, no chunking needed) case on this project's dev GPU,
with one outlier tile taking 285s for a single step (likely allocator
fragmentation/defrag stalls near the VRAM ceiling, not the chunking math
itself). This is a *capability* fix (avoids an OOM crash), not a speed one,
on hardware with no working efficient kernel - don't expect tile_num=2 to be
faster than tile_num=4 here even though it no longer crashes.

Because of that cost, chunked_sdpa always tries the real flash/efficient
kernel first (see the sdpa_kernel try/except below) and only falls back to
the manual loop if that genuinely fails for the given shape/hardware. On
hardware where flash-attention actually works (NVIDIA/CUDA), this module is
effectively a no-op passthrough - large tiles get the fast native O(N)-memory
path automatically, with none of the manual-chunking slowdown measured here.

Numerically validated against F.scaled_dot_product_attention's own output
(see conversation history) - matches to within fp16 rounding.

Inference-only: no backward pass support (this project only ever runs under
torch.inference_mode()), which is what keeps this implementation simple -
no need for the recomputation tricks a real flash-attention kernel needs to
support training.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def chunked_sdpa(query, key, value, attn_mask=None, kv_chunk_size=2048):
    """Drop-in (inference-only) replacement for
    F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask,
    dropout_p=0.0, is_causal=False) - same shapes in, same shape out, O(N)
    memory in the key/value sequence length instead of O(N^2) IF no working
    flash/memory-efficient kernel is available for this shape/hardware.

    On hardware where flash-attention actually works (e.g. NVIDIA/CUDA),
    that kernel already gives O(N) memory AND is far faster than the manual
    chunked loop below (measured ~9-11x slower on this project's ROCm dev
    GPU, which has no working flash/efficient kernel at all - see module
    docstring) - so always try it first and only fall back to manual
    chunking if it genuinely isn't available for this shape/hardware. This
    makes the whole module a no-op-but-safe fallback on hardware that
    doesn't need it, instead of unconditionally paying the slow path.
    """
    seq_len_kv = key.shape[-2]
    if seq_len_kv <= kv_chunk_size:
        # Small enough that the naive kernel's O(N^2) buffer isn't the
        # problem - skip the chunking overhead.
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
    except RuntimeError:
        pass  # no working flash/efficient kernel for this shape/hardware - fall through

    scale = query.shape[-1] ** -0.5
    out = torch.zeros_like(query, dtype=torch.float32)
    running_max = torch.full(query.shape[:-1], float("-inf"), device=query.device, dtype=torch.float32)
    running_sum = torch.zeros(query.shape[:-1], device=query.device, dtype=torch.float32)

    for start in range(0, seq_len_kv, kv_chunk_size):
        end = min(start + kv_chunk_size, seq_len_kv)
        k_chunk = key[..., start:end, :].float()
        v_chunk = value[..., start:end, :].float()

        scores = torch.matmul(query.float(), k_chunk.transpose(-2, -1)) * scale
        if attn_mask is not None:
            scores = scores + attn_mask[..., start:end].float()

        chunk_max = scores.amax(dim=-1)
        new_max = torch.maximum(running_max, chunk_max)
        exp_scores = torch.exp(scores - new_max.unsqueeze(-1))
        correction = torch.exp(running_max - new_max)

        running_sum = running_sum * correction + exp_scores.sum(dim=-1)
        out = out * correction.unsqueeze(-1) + torch.matmul(exp_scores, v_chunk)
        running_max = new_max

        del scores, exp_scores

    out = out / running_sum.unsqueeze(-1)
    return out.to(query.dtype)


class MemoryEfficientAttnProcessor:
    """Byte-for-byte copy of diffusers' AttnProcessor2_0 (attention_processor.py),
    except the single F.scaled_dot_product_attention call is replaced with
    chunked_sdpa. Drop in via unet.set_attn_processor(...) - see
    enable_chunked_attention below."""

    def __init__(self, kv_chunk_size=2048):
        self.kv_chunk_size = kv_chunk_size

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = chunked_sdpa(query, key, value, attn_mask=attention_mask, kv_chunk_size=self.kv_chunk_size)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def enable_chunked_attention(module, kv_chunk_size=2048):
    """Replace every attention layer's processor in `module` (a UNet, VAE,
    or any nn.Module with diffusers Attention submodules) with
    MemoryEfficientAttnProcessor. Call this once, right after loading the
    model, before any inference."""
    if hasattr(module, "set_attn_processor"):
        module.set_attn_processor(MemoryEfficientAttnProcessor(kv_chunk_size=kv_chunk_size))
    else:
        # Fallback for modules without the convenience method (e.g. the VAE
        # in some diffusers versions) - walk submodules and swap any
        # Attention instance's .processor directly.
        from diffusers.models.attention_processor import Attention

        for submodule in module.modules():
            if isinstance(submodule, Attention):
                submodule.processor = MemoryEfficientAttnProcessor(kv_chunk_size=kv_chunk_size)
