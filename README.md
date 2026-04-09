# VGGT Serve

`vggt-serve` is a small FastAPI server for running VGGT on a set of RGB images and returning camera poses, depth, and a point cloud.

## Quick Start With Docker

This is the easiest way to run the service.

### Requirements

- Docker
- Docker Compose
- NVIDIA Container Toolkit if you want GPU inference

### 1. Clone and initialize the submodule

```bash
git clone <your-repo-url>
cd vggt-server
git submodule update --init --recursive
```

### 2. Start the server

Choose a port. The same port is used inside the container and on the host.

```bash
scripts/docker_compose.sh up --port 9000 -d
```

Useful variants:

```bash
scripts/docker_compose.sh logs --port 9000
scripts/docker_compose.sh down
scripts/docker_compose.sh up --port 9000 --cpu
```

Notes:

- The first run can take a while because Docker builds the image.
- The first model load can also take time because VGGT weights may be downloaded.
- `--cpu` is supported, but VGGT is intended for GPU use.

### 3. Check the server

```bash
curl http://127.0.0.1:9000/healthz
curl http://127.0.0.1:9000/readyz
```

`/readyz` returns `200` when the model is loaded and ready.

### 4. Send a test request

Using the bundled Python client:

```bash
python scripts/client_example.py \
  vggt/examples/kitchen/images/00.png \
  vggt/examples/kitchen/images/01.png \
  --base-url http://127.0.0.1:9000 \
  --scene-id kitchen-demo \
  --download-dir ./client_outputs
```

Or with `curl`:

```bash
curl -X POST http://127.0.0.1:9000/v1/reconstructions \
  -F "scene_id=kitchen-demo" \
  -F "depth_conf_threshold=5.0" \
  -F "images=@vggt/examples/kitchen/images/00.png" \
  -F "images=@vggt/examples/kitchen/images/01.png"
```

## Remote Server Usage Over SSH

If the Docker container runs on a remote server, you can still use the client script from your own machine.

### 1. Start the server on the remote machine

On the server:

```bash
scripts/docker_compose.sh up --port 8080 -d
```

### 2. Forward the port to your local machine

On your local machine:

```bash
ssh -L 8080:127.0.0.1:8080 your_user@your_server
```

### 3. Run the client locally

On your local machine:

```bash
python scripts/client_example.py \
  /path/to/image_01.png \
  /path/to/image_02.png \
  --base-url http://127.0.0.1:8080 \
  --download-dir ./client_outputs
```

In this setup:

- your images stay on your local machine,
- inference runs on the remote server,
- downloaded artifacts are saved on your local machine.

## Local Conda Run

If you do not want Docker, you can run the service directly.

```bash
conda env create -f environment.yml
conda activate vggt-serve-py312
python scripts/check_env.py
uvicorn vggt_serve.app:app --host 0.0.0.0 --port 8000
```

## Main Endpoints

- `GET /healthz`
- `GET /readyz`
- `POST /v1/reconstructions`
- `GET /v1/artifacts/{request_id}/{name}`

## Server Outputs

Each successful request is stored on the server under:

```text
data/runs/<request_id>/
```

Typical files:

- `request.json`
- `result.json`
- `depth.npz`
- `point_cloud.ply`

## Docker Files

- [Dockerfile](/mnt/Backup2nd/Research/vggt-server/Dockerfile): service image
- [docker-compose.yml](/mnt/Backup2nd/Research/vggt-server/docker-compose.yml): base compose config
- [docker-compose.gpu.yml](/mnt/Backup2nd/Research/vggt-server/docker-compose.gpu.yml): GPU overlay
- [docker_compose.sh](/mnt/Backup2nd/Research/vggt-server/scripts/docker_compose.sh): simple wrapper script
