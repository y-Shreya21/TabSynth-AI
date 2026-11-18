"""Client communication interface connecting Streamlit to the FastAPI backend."""

import os
import requests
from typing import Dict, Any, List, Optional

API_BASE_URL = os.getenv("TABSYNTH_API_URL", "http://127.0.0.1:8000")


class TabSynthAPIClient:
    """REST client for TabSynth Analytics API with graceful error handling."""

    def __init__(self, base_url: str = API_BASE_URL, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_online(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def get_health(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def list_datasets(self) -> List[Dict[str, Any]]:
        r = requests.get(f"{self.base_url}/api/v1/datasets", timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("datasets", [])

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/api/v1/datasets/{dataset_id}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def upload_dataset(self, file_bytes: bytes, filename: str, name: Optional[str] = None, target_column: Optional[str] = None) -> Dict[str, Any]:
        files = {"file": (filename, file_bytes, "text/csv")}
        data = {}
        if name:
            data["name"] = name
        if target_column:
            data["target_column"] = target_column
        r = requests.post(f"{self.base_url}/api/v1/datasets/upload", files=files, data=data, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def start_training(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self.base_url}/api/v1/models/train", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def list_models(self) -> List[Dict[str, Any]]:
        r = requests.get(f"{self.base_url}/api/v1/models", timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("models", [])

    def get_job(self, job_id: str) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/api/v1/jobs/{job_id}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def list_jobs(self) -> List[Dict[str, Any]]:
        r = requests.get(f"{self.base_url}/api/v1/jobs", timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("jobs", [])

    def generate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self.base_url}/api/v1/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def evaluate_fidelity(self, real_path: str, synth_path: str, target_column: Optional[str] = None) -> Dict[str, Any]:
        payload = {
            "real_data_path": real_path,
            "synthetic_data_path": synth_path,
            "target_column": target_column,
        }
        r = requests.post(f"{self.base_url}/api/v1/evaluate/fidelity", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def evaluate_utility(self, real_train: str, synth_train: str, real_test: str, target_column: str) -> Dict[str, Any]:
        payload = {
            "real_train_path": real_train,
            "synthetic_train_path": synth_train,
            "real_test_path": real_test,
            "target_column": target_column,
        }
        r = requests.post(f"{self.base_url}/api/v1/evaluate/utility", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def evaluate_privacy(self, real_train: str, real_test: str, synth_path: str, subsample: int = 500) -> Dict[str, Any]:
        payload = {
            "real_train_path": real_train,
            "real_test_path": real_test,
            "synthetic_path": synth_path,
            "subsample_size": subsample,
        }
        r = requests.post(f"{self.base_url}/api/v1/evaluate/privacy", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def list_experiments(self) -> List[Any]:
        r = requests.get(f"{self.base_url}/api/v1/experiments", timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("experiments", [])

    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/api/v1/experiments/{experiment_id}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()


client = TabSynthAPIClient()
