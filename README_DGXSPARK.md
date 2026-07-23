# DGX Spark Docker Setup for RLDX-1 and WorldDreamer

This bundle targets DGX Spark `linux/arm64` and the local repositories:

- RLDX-1: `/home/edgexpert01/projects/RLDX-1`
- RoboCasa365 source: `/home/edgexpert01/projects/robocasa`
- WorldDreamer: `/home/edgexpert01/projects/world_dreamer_server-robocasa365-multi_task`

Checkpoints, datasets, RoboCasa assets, logs, videos, and Hugging Face cache are runtime mounts. They are not copied into the images.

## Build

```bash
cd /home/edgexpert01/Downloads/dgxspark_docker_bundle/dgxspark_docker

docker buildx build --platform linux/arm64 \
  -f Dockerfile.rldx1 \
  -t rldx1-dgxspark:dev \
  /home/edgexpert01/projects

docker buildx build --platform linux/arm64 \
  -f Dockerfile.worlddreamer \
  -t worlddreamer-dgxspark:dev \
  /home/edgexpert01/projects
```

Use `INSTALL_FLASH_ATTN=1` only after confirming the container has a CUDA toolkit/NVCC that can compile for GB10. The default is `0` and both images use SDPA/eager paths where possible.

```bash
docker buildx build --platform linux/arm64 \
  --build-arg INSTALL_FLASH_ATTN=1 \
  -f Dockerfile.rldx1 \
  -t rldx1-dgxspark:flash \
  /home/edgexpert01/projects
```

## Compose

```bash
cd /home/edgexpert01/Downloads/dgxspark_docker_bundle/dgxspark_docker
docker compose -f docker-compose.dgxspark.yml config
docker compose -f docker-compose.dgxspark.yml build rldx1
docker compose -f docker-compose.dgxspark.yml build worlddreamer
```

RLDX 1-episode smoke:

```bash
RLDX_MODEL_PATH=/checkpoints/RLDX-1-FT-RC365 \
RLDX_TASK_NAME=PickPlaceCounterToCabinet \
docker compose -f docker-compose.dgxspark.yml run --rm rldx1
```

WorldDreamer composite-unseen 1-episode smoke:

```bash
WMD_MODEL_PATH=/checkpoints/world_dreamer-robocasa365-multi_task \
WMD_TASK_SET=composite_unseen \
docker compose -f docker-compose.dgxspark.yml run --rm worlddreamer
```

## GPU Check

```bash
docker run --rm --gpus all rldx1-dgxspark:dev \
  /opt/venvs/rldx/bin/python - <<'PY'
import platform, torch
print("machine", platform.machine())
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("name", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
```

Expected on DGX Spark: `machine` is `aarch64` or `arm64`, CUDA is available, and the GPU is GB10-class.

## ARM64 Check

```bash
docker image inspect rldx1-dgxspark:dev --format '{{.Architecture}}'
docker image inspect worlddreamer-dgxspark:dev --format '{{.Architecture}}'
```

Expected: `arm64`.

## MuJoCo / EGL Smoke Test

```bash
docker run --rm --gpus all \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -v /home/edgexpert01/projects/robocasa/robocasa/models/assets:/assets/robocasa:ro \
  rldx1-dgxspark:dev \
  /opt/venvs/robocasa365/bin/python - <<'PY'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import mujoco
xml = '<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>'
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
print("mujoco", mujoco.__version__, "egl smoke ok")
PY
```

## Model Load Test

RLDX:

```bash
docker run --rm --gpus all --network host --ipc host --shm-size=32g \
  -v /home/edgexpert01/projects/checkpoints:/checkpoints:ro \
  -v /home/edgexpert01/.cache/huggingface:/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  rldx1-dgxspark:dev \
  /opt/venvs/rldx/bin/python rldx/eval/run_rldx_server.py \
    --model-path /checkpoints/RLDX-1-FT-RC365 \
    --embodiment-tag GENERAL_EMBODIMENT \
    --host 127.0.0.1 --port 20000 --use-sim-policy-wrapper
```

WorldDreamer:

```bash
docker run --rm --gpus all --network host --ipc host --shm-size=32g \
  -v /home/edgexpert01/projects/checkpoints:/checkpoints:ro \
  -v /home/edgexpert01/.cache/huggingface:/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  worlddreamer-dgxspark:dev \
  /opt/venvs/worlddreamer/bin/python -m wmd_eval_runtime.model_server \
    --model-path /checkpoints/world_dreamer-robocasa365-multi_task \
    --embodiment-tag general_embodiment \
    --host 127.0.0.1 --port 20201 --device cuda
```

Stop the server after the load/ready message. These commands are not full rollouts.

## RoboCasa365 Env Creation Test

```bash
docker run --rm --gpus all \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -v /home/edgexpert01/projects/robocasa/robocasa/models/assets:/assets/robocasa:ro \
  rldx1-dgxspark:dev \
  /opt/venvs/robocasa365/bin/python - <<'PY'
import gymnasium as gym
import robocasa
env = gym.make("robocasa/PickPlaceCounterToCabinet", split="target", seed=0)
obs, info = env.reset()
print("env ok", type(obs), len(obs) if hasattr(obs, "__len__") else "no-len")
env.close()
PY
```

## Composite-Unseen 1 Episode

WorldDreamer uses the repository launcher and `configs/task_sets.yaml`:

```bash
docker compose -f docker-compose.dgxspark.yml run --rm \
  -e WMD_MODEL_PATH=/checkpoints/world_dreamer-robocasa365-multi_task \
  -e WMD_TASK_SET=composite_unseen \
  worlddreamer
```

RLDX official single-task smoke:

```bash
docker compose -f docker-compose.dgxspark.yml run --rm \
  -e RLDX_MODEL_PATH=/checkpoints/RLDX-1-FT-RC365 \
  -e RLDX_TASK_NAME=PickPlaceCounterToCabinet \
  rldx1
```

## ARM64 / GB10 Blockers to Verify

- `RLDX-1/pyproject.toml` pins `flash-attn==2.7.4.post1`. There is no guarantee of an aarch64 GB10 wheel. Dockerfiles default to `INSTALL_FLASH_ATTN=0` and `RLDX_ATTN_IMPL=sdpa`; source build is explicit via `INSTALL_FLASH_ATTN=1`.
- `RLDX-1/pixi.toml` documents Blackwell source builds for `flash-attn` and CUDA 12.8/13.0, but its workspace platforms are `linux-64`, not `linux-aarch64`; do not assume pixi lock reuse on DGX Spark ARM64.
- RLDX custom CUDA/Triton kernels live under `RLDX-1/rldx/inference/**/kernels/*.py`; accelerated inference paths may still require a CUDA toolkit that understands GB10. The compose smoke uses uncompiled/eager settings.
- Previous local smoke on GB10 hit `nvrtc: invalid value for --gpu-architecture` in CUDA shape reductions. The compose environment sets `RLDX_DISABLE_CUDA_JIT_REDUCTIONS=1`, but this depends on the current RLDX source containing that guard.
- RoboCasa365 source `/home/edgexpert01/projects/robocasa/setup.py` requires `numpy==2.2.5` and `mujoco==3.3.1`; RLDX model deps use `numpy==1.26.4`. Dockerfiles use separate venvs to avoid mixing them.
- Native aarch64 libraries observed in the local RoboCasa365 venv include MuJoCo extensions, OpenCV, PyAV, h5py, pyarrow, and numba under `RLDX-1/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/lib/python3.10/site-packages/*.so`. Do not copy that venv into an image built for a different Python/CUDA base.

## Verification Status

The Docker files have been prepared for reproducible build/run commands. Items that require Docker daemon, NGC image pull, GPU runtime, checkpoint availability, or RoboCasa assets must be verified on the DGX Spark host:

- Dockerfile syntax: run `docker buildx build --check ...` if supported by the installed Docker version.
- Compose syntax: run `docker compose -f docker-compose.dgxspark.yml config`.
- Python import smoke: run the GPU/MuJoCo/RoboCasa commands above.
- Model load and rollout are intentionally not run by this README.
