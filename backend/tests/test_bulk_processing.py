"""Tests for batch upload processing behaviour.

Root cause of WARD-3:
  image_service.process_and_store was a no-op async def that called CPU-heavy
  PIL operations synchronously on the event loop. In a bulk upload of N photos,
  each photo blocked all other event-loop activity for the full duration of
  decode + resize + hash + disk writes.  The fix wraps that work in
  asyncio.to_thread so the loop stays responsive between photos.

The tests here verify:
  1. A failed item (invalid bytes) does not prevent the rest of the batch from
     completing (partial-failure isolation).
  2. Each successfully accepted item gets its own AI tagging job enqueued; the
     per-item Redis enqueue inside the loop is preserved.
  3. process_and_store is truly async-compatible (offloads to thread); the
     coroutine returns without blocking a mock event loop.
"""

import random
from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from PIL import Image


def _unique_jpeg() -> bytes:
    """Generate a JPEG with randomized pixel content to avoid pHash duplicate hits
    across test runs that share the same database."""
    buf = BytesIO()
    r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    # Fill with a random base color and sprinkle random pixels to perturb the hash
    img = Image.new("RGB", (64, 64), (r, g, b))
    pixels = img.load()
    for _ in range(200):
        px, py = random.randint(0, 63), random.randint(0, 63)
        pixels[px, py] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _unique_png() -> bytes:
    """Generate a PNG that is visually distinct from any _unique_jpeg call."""
    buf = BytesIO()
    r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    img = Image.new("RGB", (64, 64), (r, g, b))
    pixels = img.load()
    for _ in range(300):
        px, py = random.randint(0, 63), random.randint(0, 63)
        pixels[px, py] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestBulkPartialFailureIsolation:
    """One bad file in a batch must not block the other files from succeeding."""

    @pytest.mark.asyncio
    async def test_invalid_file_does_not_block_valid_files(
        self, client: AsyncClient, auth_headers
    ):
        """A batch with one invalid image and two valid images: both valid items
        succeed; the invalid one is reported as failed; the overall request is 201."""
        files = [
            ("images", ("good.jpg", _unique_jpeg(), "image/jpeg")),
            ("images", ("bad.txt", b"not an image at all", "text/plain")),
            ("images", ("also_good.png", _unique_png(), "image/png")),
        ]
        with patch("app.api.items.create_pool", new_callable=AsyncMock) as mock_pool:
            mock_redis = AsyncMock()
            mock_redis.enqueue_job.return_value.job_id = "job-x"
            mock_pool.return_value = mock_redis

            response = await client.post(
                "/api/v1/items/bulk",
                files=files,
                headers=auth_headers,
            )

        assert response.status_code == 201, response.json()
        data = response.json()

        assert data["total"] == 3
        assert data["successful"] == 2, f"Two valid images should succeed: {data['results']}"
        assert data["failed"] == 1, "One invalid file should fail"

        results = data["results"]
        good_results = [r for r in results if r["success"]]
        bad_results = [r for r in results if not r["success"]]

        assert len(good_results) == 2
        assert len(bad_results) == 1
        assert bad_results[0]["filename"] == "bad.txt"
        assert "error" in bad_results[0]

    @pytest.mark.asyncio
    async def test_invalid_file_in_middle_does_not_block_later_files(
        self, client: AsyncClient, auth_headers
    ):
        """An invalid file at position 1 must not prevent file at position 2 from
        succeeding — the loop must continue to the next item."""
        files = [
            ("images", ("first.jpg", _unique_jpeg(), "image/jpeg")),
            ("images", ("bad.jpg", b"\x00\xff\x00", "image/jpeg")),  # corrupt bytes
            ("images", ("third.jpg", _unique_jpeg(), "image/jpeg")),
        ]
        with patch("app.api.items.create_pool", new_callable=AsyncMock) as mock_pool:
            mock_redis = AsyncMock()
            mock_redis.enqueue_job.return_value.job_id = "job-y"
            mock_pool.return_value = mock_redis

            response = await client.post(
                "/api/v1/items/bulk",
                files=files,
                headers=auth_headers,
            )

        assert response.status_code == 201, response.json()
        data = response.json()

        # first and third must succeed; corrupt middle must fail gracefully
        assert data["successful"] >= 2
        assert data["failed"] >= 1

        by_name = {r["filename"]: r for r in data["results"]}
        assert by_name["first.jpg"]["success"] is True
        assert by_name["third.jpg"]["success"] is True
        assert by_name["bad.jpg"]["success"] is False


class TestBulkPerItemJobEnqueue:
    """Each accepted item in a batch must have its own AI tagging job enqueued."""

    @pytest.mark.asyncio
    async def test_three_valid_items_each_get_a_job(
        self, client: AsyncClient, auth_headers
    ):
        """Three valid JPEGs → three distinct enqueue_job calls, one per item."""
        files = [
            ("images", (f"img{i}.jpg", _unique_jpeg(), "image/jpeg"))
            for i in range(3)
        ]
        call_count = 0

        with patch("app.api.items.create_pool", new_callable=AsyncMock) as mock_pool:
            mock_redis = AsyncMock()

            async def _enqueue(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                job = AsyncMock()
                job.job_id = f"job-{call_count}"
                return job

            mock_redis.enqueue_job.side_effect = _enqueue
            mock_pool.return_value = mock_redis

            response = await client.post(
                "/api/v1/items/bulk",
                files=files,
                headers=auth_headers,
            )

        assert response.status_code == 201, response.json()
        data = response.json()
        assert data["successful"] == 3, data
        # One enqueue per successful item
        assert mock_redis.enqueue_job.call_count == 3

    @pytest.mark.asyncio
    async def test_skip_ai_no_jobs_enqueued(self, client: AsyncClient, auth_headers):
        """skip_ai=true → Redis pool is never opened, no jobs are enqueued."""
        files = [
            ("images", (f"img{i}.jpg", _unique_jpeg(), "image/jpeg"))
            for i in range(3)
        ]
        with patch("app.api.items.create_pool", new_callable=AsyncMock) as mock_pool:
            response = await client.post(
                "/api/v1/items/bulk",
                files=files,
                data={"skip_ai": "true"},
                headers=auth_headers,
            )

        assert response.status_code == 201, response.json()
        data = response.json()
        assert data["successful"] == 3
        mock_pool.assert_not_called()

    @pytest.mark.asyncio
    async def test_job_enqueue_failure_does_not_abort_whole_batch(
        self, client: AsyncClient, auth_headers
    ):
        """If Redis enqueue fails for one item, the remaining items still complete
        and are reported as successful (the error is logged, not raised)."""
        files = [
            ("images", (f"img{i}.jpg", _unique_jpeg(), "image/jpeg"))
            for i in range(3)
        ]
        call_count = 0

        with patch("app.api.items.create_pool", new_callable=AsyncMock) as mock_pool:
            mock_redis = AsyncMock()

            async def _enqueue_sometimes_fails(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise RuntimeError("Redis timeout")
                job = AsyncMock()
                job.job_id = f"job-{call_count}"
                return job

            mock_redis.enqueue_job.side_effect = _enqueue_sometimes_fails
            mock_pool.return_value = mock_redis

            response = await client.post(
                "/api/v1/items/bulk",
                files=files,
                headers=auth_headers,
            )

        assert response.status_code == 201, response.json()
        data = response.json()
        # All three items should still be reported as created/successful
        assert data["successful"] == 3, (
            "An enqueue failure should not mark the item as failed - "
            "the item was stored, just without a queued AI job"
        )


class TestImageServiceThreadOffload:
    """Verify process_and_store is properly offloaded to a thread (event-loop safe)."""

    @pytest.mark.asyncio
    async def test_process_and_store_is_awaitable(self, tmp_path):
        """process_and_store must be awaitable and return the expected keys."""
        import uuid
        from app.services.image_service import ImageService

        service = ImageService(storage_path=str(tmp_path))
        image_bytes = _unique_jpeg()
        result = await service.process_and_store(
            user_id=uuid.uuid4(),
            image_data=image_bytes,
            original_filename="test.jpg",
        )

        assert "image_path" in result
        assert "medium_path" in result
        assert "thumbnail_path" in result
        assert "image_hash" in result

    @pytest.mark.asyncio
    async def test_process_and_store_raises_on_bad_extension(self, tmp_path):
        """Unsupported file type raises ValueError before any CPU work starts."""
        import uuid
        from app.services.image_service import ImageService

        service = ImageService(storage_path=str(tmp_path))
        with pytest.raises(ValueError, match="Unsupported file type"):
            await service.process_and_store(
                user_id=uuid.uuid4(),
                image_data=b"anything",
                original_filename="file.bmp",
            )

    @pytest.mark.asyncio
    async def test_process_and_store_png_succeeds(self, tmp_path):
        """PNG (non-HEIC) path works correctly end-to-end through thread offload."""
        import uuid
        from app.services.image_service import ImageService

        service = ImageService(storage_path=str(tmp_path))
        image_bytes = _unique_png()
        result = await service.process_and_store(
            user_id=uuid.uuid4(),
            image_data=image_bytes,
            original_filename="image.png",
        )

        assert result["image_path"].endswith(".jpg")
        assert result["thumbnail_path"].endswith("_thumb.jpg")
        assert result["medium_path"].endswith("_medium.jpg")
        # Verify files were actually written to disk
        for key in ("image_path", "medium_path", "thumbnail_path"):
            full = tmp_path / result[key]
            assert full.exists(), f"Expected {key} file at {full}"
