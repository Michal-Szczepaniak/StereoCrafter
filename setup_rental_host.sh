#!/usr/bin/env bash
# One-shot environment setup for a freshly-rented CUDA GPU box (vast.ai,
# RunPod, etc). Run this AFTER cloning the repo (this script doesn't clone
# itself - it assumes it's already sitting in the repo root):
#
#   git clone --recursive git@github.com:Michal-Szczepaniak/StereoCrafter.git
#   cd StereoCrafter
#   HF_TOKEN=hf_xxx ./setup_rental_host.sh
#
# HF_TOKEN must be a Hugging Face access token (Settings -> Access Tokens,
# type "Read", with "Read access to contents of all public gated repos you
# can access" checked) - required because stable-video-diffusion-img2vid-xt-1-1
# is gated. You must ALSO have already clicked "Agree and access repository"
# on that model's HF page at least once with the account the token belongs
# to - that part can't be scripted. DepthCrafter/StereoCrafter weights are
# open, no token needed for those.
#
# Every step below fixes a real failure hit while setting this up live on a
# vast.ai "Nvidia CUDA" template box - see each step's comment for what broke
# and why. Safe to re-run if it dies partway through (conda/pip/git steps are
# all idempotent).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

: "${HF_TOKEN:?Set HF_TOKEN to a Hugging Face read token first (needed for the gated SVD weights) - see the header comment above.}"
: "${HF_USERNAME:?Set HF_USERNAME to your Hugging Face username first.}"

echo "==================================================================="
echo "1/5: Forward-Warp - swap ROCm/HIP build files back to CUDA"
echo "==================================================================="
# The DepthCrafter/Forward-Warp submodules are forked+patched for our own
# ROCm dev box (see .gitmodules). Forward-Warp's patch specifically rewrote
# its build to target HIP (CppExtension instead of CUDAExtension, <hip/
# hip_runtime.h> instead of <cuda.h>) - that fails to compile on a real CUDA
# box (no HIP toolchain). The only patches on stereocrafter-patches are
# build-system changes, nothing algorithmic, so on CUDA hardware we just
# pull the two build files back from upstream instead.
(
    cd dependency/Forward-Warp
    rm -rf Forward_Warp/cuda/build
    git fetch origin
    git checkout origin/master -- Forward_Warp/cuda/setup.py Forward_Warp/cuda/forward_warp_cuda_kernel.cu
)

echo
echo "==================================================================="
echo "2/5: conda env + pinned requirements"
echo "==================================================================="
# requirements.txt pins old versions (torch==2.0.1, xformers==0.0.20,
# decord==0.6.0) unlikely to have wheels for a brand-new interpreter -
# python=3.10 is a safe middle ground.
if ! conda env list | grep -q "^stereocrafter "; then
    conda create -n stereocrafter python=3.10 -y
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
# conda's own activate.d/deactivate.d hooks (e.g. the one gxx_linux-64 installs
# below, in step 3) aren't written to be safe under `set -u` - they reference
# backup vars like CONDA_BACKUP_CXX that are only set in some code paths, and
# blow up ("unbound variable") under nounset. conda activate/install/remove
# from here on can trigger those hooks, so nounset gets turned off for the
# rest of the script rather than fighting conda's scripts one at a time.
set +u
conda activate stereocrafter

pip install -r requirements.txt

# setuptools>=82 (Feb 2026) removed the pkg_resources module entirely.
# torch==2.0.1's own torch/utils/cpp_extension.py still does
# `from pkg_resources import packaging` - without this pin, building ANY
# CUDA extension (Forward-Warp, below) fails with
# "ModuleNotFoundError: No module named 'pkg_resources'".
pip install "setuptools<82"

echo
echo "==================================================================="
echo "3/5: CUDA 11.x toolkit (matches torch==2.0.1's bundled version)"
echo "==================================================================="
# torch==2.0.1's plain PyPI wheel was compiled against CUDA 11.7. Rental
# box images ship whatever system-wide nvcc happens to be current (seen:
# 12.8 on one box, 13.3 on another) for general use, and torch's extension
# builder HARD-fails when the major version differs from what it was built
# with: "The detected CUDA version (X) mismatches the version that was used
# to compile PyTorch (11.7)." A minor-version gap (e.g. 11.7 vs 11.8) is
# only a warning, not fatal - so any 11.x toolkit clears this, we don't need
# an exact match. Installing it INSIDE the conda env (rather than touching
# the system-wide CUDA install) fixes the build without affecting anything
# else - $CONDA_PREFIX/bin is already ahead of the system CUDA on PATH once
# the env is active, and the GPU driver supports newer CUDA runtimes than
# what it's asked to run, so an 11.x-compiled extension still runs fine
# regardless of how new the system toolkit is.
current_major="$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' || echo 0)"
if [[ "$current_major" != "11" ]]; then
    # NOTE (learned the hard way): the `cuda-toolkit=X.Y` metapackage's own
    # version pin does NOT reliably cascade to its components on nvidia's
    # channel - `cuda-toolkit=11.7.1` was observed pulling in `cuda-nvcc
    # 13.3.73` (whatever the channel's current release is) instead of an
    # actual 11.7 build. Target the compiler sub-package directly instead,
    # which does respect the pin.
    conda remove cuda-nvcc cuda-toolkit -y 2>/dev/null || true
    # cuda-nvcc alone isn't enough - `conda remove cuda-toolkit` above also
    # cascades away every dev sub-package that was only pulled in as the
    # metapackage's dependency: cuda-cudart-dev (cuda_runtime.h - "fatal
    # error: cuda_runtime.h: No such file or directory") and cuda-cccl
    # (thrust/cub headers - "fatal error: thrust/complex.h: No such file or
    # directory", needed transitively via torch's own c10/util/complex.h).
    # Pin all three explicitly up front rather than whack-a-moling each
    # missing header one build attempt at a time.
    conda install -c nvidia -c conda-forge cuda-nvcc=11.7 cuda-cudart-dev=11.7 cuda-cccl=11.7 -y \
        || conda install -c nvidia -c conda-forge cuda-nvcc=11.8 cuda-cudart-dev=11.8 cuda-cccl=11.8 -y
    # `conda remove` above can cascade-drop the C++ compiler as an orphaned
    # dependency (hit live as `subprocess.CalledProcessError` from `which
    # x86_64-conda-linux-gnu-c++` during the build below) - make sure it's
    # there explicitly rather than hoping it survived. Pinned to major
    # version 11: CUDA 11.7's nvcc hard-caps the host compiler at <12.0 and
    # errors out ("current installed version ... is greater than the
    # maximum required version") on whatever conda-forge's current default
    # (16.1 seen live) resolves to unpinned.
    conda install -c conda-forge "gxx_linux-64=11" "gcc_linux-64=11" -y
    hash -r
fi
export CUDA_HOME="$CONDA_PREFIX"
echo "==> using nvcc: $(which nvcc)"
nvcc --version
which x86_64-conda-linux-gnu-c++ >/dev/null || { echo "==> ERROR: C++ compiler still missing after install." >&2; exit 1; }
installed_major="$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' || echo 0)"
if [[ "$installed_major" != "11" ]]; then
    echo "==> ERROR: nvcc is still not CUDA 11.x after install (got major version $installed_major)." >&2
    echo "    Check 'conda search -c nvidia -c conda-forge \"cuda-toolkit=11.*\"' for what's actually available." >&2
    exit 1
fi

# The linker needs an unversioned libcudart.so for -lcudart, but conda's
# own cuda-cudart package keeps landing on whatever the channel's current
# release is (13.x) regardless of an explicit =11.7 pin - same resolution
# weirdness as cuda-toolkit above, confirmed live even after an explicit
# `conda install cuda-cudart=11.7`. Skip fighting conda for this one: pip's
# nvidia-cuda-runtime-cu11 (already pulled in automatically as torch's own
# dependency) ships the correct 11.0 lib, just without the unversioned
# symlink -lcudart looks for. Symlink it and point the linker there
# directly - guaranteed correct version since torch declared it itself.
# NOTE: nvidia.cuda_runtime is a namespace package (no __init__.py), so
# __file__ is None on it - use __path__ instead, which namespace packages do
# have.
nvidia_cudart_dir="$(python -c "import nvidia.cuda_runtime, os; print(os.path.join(list(nvidia.cuda_runtime.__path__)[0], 'lib'))")"
if [[ -f "$nvidia_cudart_dir"/libcudart.so.11.* && ! -e "$nvidia_cudart_dir/libcudart.so" ]]; then
    ln -sf "$(ls "$nvidia_cudart_dir"/libcudart.so.11.* | head -1)" "$nvidia_cudart_dir/libcudart.so"
fi
export LIBRARY_PATH="$nvidia_cudart_dir:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$nvidia_cudart_dir:${LD_LIBRARY_PATH:-}"

echo
echo "==================================================================="
echo "4/5: build Forward-Warp"
echo "==================================================================="
# NOTE (confirmed NOT an issue live, despite the concern below being
# reasonable on paper): the vanilla upstream kernel's use of the deprecated
# .type() API only produced deprecation warnings ("Tensor.data<T>() is
# deprecated... use Tensor.data_ptr<T>()"), not a compile error, against
# torch==2.0.1. Left the escape hatch below in case a different torch
# version ever does hard-remove it.
# If this fails with something like "no member named 'type' in
# 'at::Tensor'" (AT_DISPATCH_FLOATING_TYPES(im0.type(), ...)) - the vanilla
# upstream kernel we restored in step 1 uses the old .type() API, which may
# have been dropped from newer PyTorch's ATen headers. Fix: in
# dependency/Forward-Warp/Forward_Warp/cuda/forward_warp_cuda_kernel.cu,
# replace the 3 occurrences of `.type()` with `.scalar_type()` (that's the
# exact fix the ROCm/HIP fork already made, for the same reason) and rerun
# this step. Not applied automatically here since it wasn't confirmed
# necessary against every torch version - only patch it if you actually hit
# this error.
(
    cd dependency/Forward-Warp
    chmod +x install.sh
    cd Forward_Warp/cuda
    python setup.py install
    cd ../..
    python setup.py install
)
python -c "import Forward_Warp, forward_warp_cuda; print('==> Forward-Warp OK')"

echo
echo "==================================================================="
echo "5/5: model weights"
echo "==================================================================="
mkdir -p weights
cd weights
git lfs install

if [[ ! -d stable-video-diffusion-img2vid-xt-1-1 ]]; then
    git clone "https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1"
fi
if [[ ! -d DepthCrafter ]]; then
    git clone https://huggingface.co/tencent/DepthCrafter
fi
if [[ ! -d StereoCrafter ]]; then
    git clone https://huggingface.co/TencentARC/StereoCrafter
fi
cd ..

echo
echo "==================================================================="
echo "Sanity checks"
echo "==================================================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
du -sh weights/*

echo
echo "==> Setup complete. Next: copy the source video over, then run a"
echo "    smoke test before the full episode - e.g.:"
echo "      source presets/24GB.env"
echo "      PROCESS_LENGTH=\$CHUNK_SIZE ./run_stereo.sh source_video/whatever.mkv"
