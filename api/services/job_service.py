"""Asynchronous background job management service with SQLite persistence and concurrency controls."""

import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Optional, Any, Callable
from api.config import settings
from api.db.models import JobRecord
from api.db.repositories import JobRepository

logger = logging.getLogger("tabsynth.jobs")


class JobManager:
    """Manages background task execution, concurrency bounds, and persistent state transitions."""

    _instance: Optional["JobManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, max_workers: int = settings.max_concurrent_jobs):
        if self._initialized:
            return
        self.max_workers = max(1, max_workers)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="TabSynthJobWorker")
        self.semaphore = threading.Semaphore(self.max_workers)
        self._cancel_events: Dict[str, threading.Event] = {}
        self._futures: Dict[str, Future] = {}
        self._initialized = True
        logger.info(f"Initialized JobManager with {self.max_workers} worker slot(s).")

    def submit_job(
        self,
        job_type: str,
        task_fn: Callable[..., Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        total_steps: int = 100,
        *args,
        **kwargs,
    ) -> str:
        """Submit a background job for asynchronous execution."""
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        cancel_event = threading.Event()
        self._cancel_events[job_id] = cancel_event

        record = JobRecord(
            id=job_id,
            job_type=job_type,
            status="queued",
            progress=0.0,
            current_step=0,
            total_steps=total_steps,
            metadata=metadata or {},
        )
        JobRepository.create(record)

        future = self.executor.submit(
            self._execute_job,
            job_id=job_id,
            task_fn=task_fn,
            cancel_event=cancel_event,
            total_steps=total_steps,
            args=args,
            kwargs=kwargs,
        )
        self._futures[job_id] = future
        logger.info(f"Enqueued job {job_id} of type '{job_type}'.")
        return job_id

    def _execute_job(
        self,
        job_id: str,
        task_fn: Callable,
        cancel_event: threading.Event,
        total_steps: int,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Worker wrapper managing concurrency, progress callbacks, and completion."""
        with self.semaphore:
            if cancel_event.is_set():
                JobRepository.update_progress(
                    job_id=job_id,
                    status="cancelled",
                    progress=0.0,
                    current_step=0,
                    total_steps=total_steps,
                    error="Job cancelled while queued.",
                )
                logger.info(f"Job {job_id} was cancelled before execution.")
                return

            # Mark running
            JobRepository.update_progress(
                job_id=job_id,
                status="running",
                progress=0.0,
                current_step=0,
                total_steps=total_steps,
            )
            logger.info(f"Started executing job {job_id}.")

            def update_progress(current_step: int, extra_result: Optional[Dict[str, Any]] = None):
                prog = min(1.0, max(0.0, current_step / max(1, total_steps)))
                JobRepository.update_progress(
                    job_id=job_id,
                    status="running",
                    progress=prog,
                    current_step=current_step,
                    total_steps=total_steps,
                    result=extra_result,
                )

            def is_cancelled() -> bool:
                return cancel_event.is_set()

            try:
                result = task_fn(
                    *args,
                    job_id=job_id,
                    progress_callback=update_progress,
                    is_cancelled_fn=is_cancelled,
                    **kwargs,
                )

                if cancel_event.is_set():
                    JobRepository.update_progress(
                        job_id=job_id,
                        status="cancelled",
                        progress=1.0,
                        current_step=total_steps,
                        error="Job was safely cancelled during execution.",
                    )
                    logger.info(f"Job {job_id} cancelled during execution.")
                else:
                    JobRepository.update_progress(
                        job_id=job_id,
                        status="completed",
                        progress=1.0,
                        current_step=total_steps,
                        result=result,
                    )
                    logger.info(f"Successfully finished job {job_id}.")

            except Exception as exc:
                logger.exception(f"Exception encountered in job {job_id}: {exc}")
                JobRepository.update_progress(
                    job_id=job_id,
                    status="failed",
                    progress=0.0,
                    current_step=0,
                    total_steps=total_steps,
                    error=str(exc),
                )
            finally:
                self._cancel_events.pop(job_id, None)
                self._futures.pop(job_id, None)

    def cancel_job(self, job_id: str) -> bool:
        """Signals cancellation to an active or queued job."""
        event = self._cancel_events.get(job_id)
        if event:
            event.set()
            JobRepository.update_progress(
                job_id=job_id,
                status="cancelled",
                progress=0.0,
                current_step=0,
                total_steps=0,
                error="Cancellation requested by user.",
            )
            logger.info(f"Signal sent to cancel job {job_id}.")
            return True

        # Check if exists in DB
        job = JobRepository.get_by_id(job_id)
        if job and job.status in ["queued", "running"]:
            JobRepository.update_progress(
                job_id=job_id,
                status="cancelled",
                progress=job.progress,
                current_step=job.current_step,
                total_steps=job.total_steps,
                error="Job marked cancelled in database.",
            )
            return True
        return False

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return JobRepository.get_by_id(job_id)

    def list_jobs(self, limit: int = 50) -> List[JobRecord]:
        return JobRepository.list_all(limit=limit)


# Global singleton manager instance
job_manager = JobManager()
