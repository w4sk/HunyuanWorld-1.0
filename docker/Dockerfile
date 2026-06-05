# syntax=docker/dockerfile:1.7
# HunyuanWorld-1.0 — reproducible GPU container
# Base: CUDA 12.4 *devel* image provides nvcc 12.4, required to build
# pytorch3d / flash-attn / natten from source against torch 2.5.0+cu124.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    # 4090 is Ada (sm_89). Restrict arch so source builds are fast & low-memory.
    TORCH_CUDA_ARCH_LIST="8.9" \
    FORCE_CUDA=1 \
    # Cap compile parallelism: host has 31GB RAM, nvcc jobs are memory-heavy.
    MAX_JOBS=4 \
    # Make pip resilient to stalled CDN connections (saw a 50min hang otherwise).
    PIP_DEFAULT_TIMEOUT=60 \
    PIP_RETRIES=10 \
    PIP_NO_INPUT=1

# ---- System packages (build tools + OpenGL/ffmpeg runtime for cv2/open3d) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates build-essential \
        cmake ninja-build \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---- Miniconda ----
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && conda config --set always_yes true

# Recent conda refuses to solve the Anaconda default channels until their
# Terms of Service are accepted non-interactively.
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ---- Conda environment: conda packages ONLY ----
# We deliberately split the official HunyuanWorld.yaml into conda-only deps
# (here) + a pip requirements.txt (below). conda's built-in pip phase buffered
# all output and hung indefinitely on a wedged PyPI/Fastly connection; running
# pip ourselves gives live progress and reliable --timeout/--retries.
# Regenerate both from HunyuanWorld.yaml with docker/split-env.py if the spec changes.
COPY docker/conda-base.yaml /tmp/conda-base.yaml
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    conda env create -f /tmp/conda-base.yaml

# All subsequent RUN steps execute inside the HunyuanWorld env.
SHELL ["conda", "run", "--no-capture-output", "-n", "HunyuanWorld", "/bin/bash", "-c"]

# ---- Pip dependencies (our own command: visible progress + honored timeouts) ----
COPY docker/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --timeout 60 --retries 10 --progress-bar off -r /tmp/requirements.txt

# ---- Heavy compiled / git deps ----
# Split out of the conda env so each is its own cacheable layer and a failure
# in one slow build doesn't invalidate the whole environment.
# pytorch3d: CUDA source build against torch 2.5 + cu124 (slowest step).
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# flash-attn: pulls the matching prebuilt wheel from GH releases when ABI matches,
# else compiles (MAX_JOBS bounds memory).
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-build-isolation flash-attn==2.7.4.post1
# natten + sageattention: small CUDA source builds.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install natten==0.14.4
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install sageattention==1.0.6
# MoGe depth model (pure python).
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install "git+https://github.com/microsoft/MoGe.git"

# ---- Real-ESRGAN (super-resolution) ----
RUN git clone https://github.com/xinntao/Real-ESRGAN.git /opt/Real-ESRGAN \
    && cd /opt/Real-ESRGAN \
    && pip install basicsr-fixed facexlib gfpgan \
    && pip install -r requirements.txt \
    && python setup.py develop \
    # requirements.txt pulls the broken upstream `basicsr`; force the patched fork to win.
    && pip install --no-deps --force-reinstall basicsr-fixed

# ---- ZIM Anything (matting) — install the python package only.
# Checkpoints (~1.4GB) live on the host at ./ZIM/zim_vit_l_2092 (bind-mounted),
# because the code loads them via the CWD-relative path "./ZIM/zim_vit_l_2092".
RUN git clone https://github.com/naver-ai/ZIM.git /opt/ZIM \
    && cd /opt/ZIM \
    && pip install -e .

# ---- Draco (mesh compression, optional .drc export) ----
RUN git clone https://github.com/google/draco.git /opt/draco \
    && cd /opt/draco && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make -j${MAX_JOBS} \
    && make install \
    && ldconfig

# ---- Pin huggingface_hub last ----
# Installing MoGe (HEAD = 2.0.0) drags in gradio/fastapi and bumps huggingface_hub
# to 1.x, which breaks transformers 4.51.0 / tokenizers (both need hub <1.0).
# MoGe itself works fine with 0.30.2, so we force it back as the final step.
# (gradio's >=0.33.5 request stays unsatisfied but gradio is unused by the CLI demos.)
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install "huggingface_hub==0.30.2"

# ---- utils3d 1.3 compat patches (it ships with MoGe HEAD but its API/numpy
# assumptions drift from what HunyuanWorld's pano_depth_utils.py expects) ----
# 1) numpy-2.0 ndarray `.mT` -> swapaxes (stack is pinned to numpy 1.24).
#    (torch-side .mT is valid on tensors and is left untouched.)
# 2) re-add `image_uv`, which 1.3 dropped (HunyuanWorld calls it for pano rays).
RUN U=/opt/conda/envs/HunyuanWorld/lib/python3.10/site-packages/utils3d/numpy \
    && sed -i 's/\.mT\b/.swapaxes(-1, -2)/g' "$U/transforms.py" \
    && printf '\n# compat shim: utils3d 1.3 dropped image_uv (HunyuanWorld needs it)\ndef image_uv(height=None, width=None, **_kwargs):\n    import numpy as _np\n    u = (_np.arange(width) + 0.5) / width\n    v = (_np.arange(height) + 0.5) / height\n    u, v = _np.meshgrid(u, v)\n    return _np.stack([u, v], axis=-1).astype(_np.float32)\n' >> "$U/__init__.py"

WORKDIR /workspace
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
