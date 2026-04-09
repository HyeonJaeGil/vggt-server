from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from vggt_serve.app import create_app
from vggt_serve.config import Settings
from vggt_serve.errors import ServiceBusyApiError, ServiceUnavailableApiError
from vggt_serve.inference import EngineRunResult
from vggt_serve.storage import ArtifactDescriptor, PreparedImage, write_depth_artifact, write_point_cloud_ply


def _image_bytes(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    image = Image.new("RGB", size, color=color)
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class StubEngine:
    def __init__(self, *, ready: bool = True, last_error: str | None = None, busy: bool = False) -> None:
        self._ready = ready
        self._last_error = last_error
        self._busy = busy

    @property
    def device_description(self) -> str | None:
        return "stub-device" if self._ready else None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def is_ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        if self._last_error:
            raise RuntimeError(self._last_error)
        self._ready = True

    def run(
        self,
        *,
        request_id: str,
        run_dir: Path,
        images: list[PreparedImage],
        depth_conf_threshold: float,
    ) -> EngineRunResult:
        if self._last_error:
            raise ServiceUnavailableApiError(self._last_error)
        if self._busy:
            raise ServiceBusyApiError()

        depth_maps = [np.ones((image.height, image.width), dtype=np.float32) for image in images]
        depth_conf = [
            np.full((image.height, image.width), depth_conf_threshold + 1.0, dtype=np.float32) for image in images
        ]
        depth_path = run_dir / "depth.npz"
        write_depth_artifact(
            depth_path,
            [image.original_filename for image in images],
            [(image.width, image.height) for image in images],
            depth_maps,
            depth_conf,
        )

        ply_path = run_dir / "point_cloud.ply"
        points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
        colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        write_point_cloud_ply(ply_path, points, colors)

        camera_results = [
            {
                "filename": image.original_filename,
                "original_size": {"width": image.width, "height": image.height},
                "cam_from_world": np.eye(4, dtype=np.float32).tolist(),
                "intrinsics": np.eye(3, dtype=np.float32).tolist(),
            }
            for image in images
        ]

        return EngineRunResult(
            camera_results=camera_results,
            artifacts=[
                ArtifactDescriptor(
                    name=depth_path.name,
                    path=depth_path,
                    content_type="application/octet-stream",
                    size_bytes=depth_path.stat().st_size,
                ),
                ArtifactDescriptor(
                    name=ply_path.name,
                    path=ply_path,
                    content_type="application/octet-stream",
                    size_bytes=ply_path.stat().st_size,
                ),
            ],
            timings_ms={"inference": 5, "postprocess": 2, "total": 7},
        )


def _make_client(tmp_path: Path, engine: StubEngine | None = None) -> TestClient:
    settings = Settings(data_root=tmp_path / "runs")
    app = create_app(settings=settings, engine=engine or StubEngine(), load_engine_on_startup=False)
    return TestClient(app)


def test_healthz(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_not_ready(tmp_path: Path) -> None:
    client = _make_client(tmp_path, engine=StubEngine(ready=False, last_error="model load failed"))
    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert response.json()["error"] == "model load failed"


def test_reconstruction_success(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    response = client.post(
        "/v1/reconstructions",
        files=[
            ("images", ("image_a.png", _image_bytes((12, 8), (255, 0, 0)), "image/png")),
            ("images", ("image_b.png", _image_bytes((8, 12), (0, 255, 0)), "image/png")),
        ],
        data={
            "scene_id": "scene-1",
            "client_request_id": "client-123",
            "depth_conf_threshold": "4.0",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "succeeded"
    assert payload["request_id"]
    assert payload["client_request_id"] == "client-123"
    assert payload["input_summary"]["image_count"] == 2
    assert len(payload["camera_results"]) == 2
    assert [artifact["name"] for artifact in payload["artifacts"]] == [
        "depth.npz",
        "point_cloud.ply",
        "result.json",
    ]

    artifact_url = payload["artifacts"][1]["url"]
    artifact_path = urlparse(artifact_url).path
    artifact_response = client.get(artifact_path)
    assert artifact_response.status_code == 200
    assert artifact_response.content.startswith(b"ply\nformat binary_little_endian 1.0\n")


def test_reconstruction_rejects_corrupt_image(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    response = client.post(
        "/v1/reconstructions",
        files=[("images", ("broken.png", b"not-a-real-image", "image/png"))],
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "validation_error"


def test_reconstruction_rejects_unsupported_media(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    response = client.post(
        "/v1/reconstructions",
        files=[("images", ("notes.txt", b"text", "text/plain"))],
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media"


def test_reconstruction_returns_busy(tmp_path: Path) -> None:
    client = _make_client(tmp_path, engine=StubEngine(busy=True))
    response = client.post(
        "/v1/reconstructions",
        files=[("images", ("image.png", _image_bytes((10, 10), (0, 0, 255)), "image/png"))],
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_busy"

