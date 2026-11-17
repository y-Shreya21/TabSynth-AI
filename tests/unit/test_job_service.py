"""Unit tests for JobManager asynchronous execution, lifecycle states, and cancellation."""

import time
import pytest
from api.services.job_service import JobManager


def dummy_successful_task(job_id, progress_callback, is_cancelled_fn, steps=3):
    for i in range(1, steps + 1):
        if is_cancelled_fn():
            return {"status": "aborted"}
        progress_callback(i)
        time.sleep(0.01)
    return {"message": "task completed", "steps": steps}


def dummy_failing_task(job_id, progress_callback, is_cancelled_fn):
    raise RuntimeError("Simulated failure in worker task")


def dummy_cancellable_task(job_id, progress_callback, is_cancelled_fn):
    for i in range(1, 100):
        if is_cancelled_fn():
            return {"status": "stopped"}
        progress_callback(i)
        time.sleep(0.05)
    return {"status": "finished"}


def test_job_manager_successful_execution():
    manager = JobManager(max_workers=2)
    job_id = manager.submit_job(
        job_type="test_type",
        task_fn=dummy_successful_task,
        metadata={"param": "value"},
        total_steps=3,
        steps=3,
    )
    assert job_id.startswith("job_")

    # Wait for completion
    for _ in range(50):
        job = manager.get_job(job_id)
        if job and job.status == "completed":
            break
        time.sleep(0.05)

    assert job is not None
    assert job.status == "completed"
    assert job.progress == 1.0
    assert job.result["message"] == "task completed"


def test_job_manager_failure_handling():
    manager = JobManager()
    job_id = manager.submit_job(
        job_type="failing_test",
        task_fn=dummy_failing_task,
        total_steps=5,
    )

    for _ in range(50):
        job = manager.get_job(job_id)
        if job and job.status == "failed":
            break
        time.sleep(0.05)

    assert job is not None
    assert job.status == "failed"
    assert "Simulated failure" in (job.error or "")


def test_job_manager_cancellation():
    manager = JobManager()
    job_id = manager.submit_job(
        job_type="cancellable_test",
        task_fn=dummy_cancellable_task,
        total_steps=100,
    )

    time.sleep(0.05)
    cancelled = manager.cancel_job(job_id)
    assert cancelled is True

    for _ in range(50):
        job = manager.get_job(job_id)
        if job and job.status == "cancelled":
            break
        time.sleep(0.05)

    assert job is not None
    assert job.status == "cancelled"
