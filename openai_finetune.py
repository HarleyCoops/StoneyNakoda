import os
import json
import time
import logging
import io
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence
from dotenv import load_dotenv
from openai import OpenAI

from scripts.validate_source_manifest import assert_paths_allowed_for_use
from stoney_config import load_stoney_config

try:
    from huggingface_hub import HfApi
except ImportError:  # pragma: no cover - optional dependency guard
    HfApi = None  # type: ignore[assignment]

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency guard
    wandb = None  # type: ignore[assignment]

# Set up logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

STATUS_TO_INDEX: Dict[str, int] = {
    "validating_files": -1,
    "queued": 0,
    "running": 1,
    "succeeded": 2,
    "failed": -2,
    "cancelled": -3,
    "expired": -4,
}


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class HuggingFacePublishConfig:
    """Resolved Hugging Face publishing configuration."""

    enabled: bool
    token: Optional[str]
    repo_id: Optional[str]
    private: bool
    allow_public_upload: bool


def _env_flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    """Parse a boolean environment variable with explicit true/false values."""

    raw = env.get(name)
    if raw is None or raw == "":
        return default
    normalized = raw.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}")


def build_hf_publish_config(env: Mapping[str, str] | None = None) -> HuggingFacePublishConfig:
    """Resolve safe Hugging Face publishing defaults from environment variables.

    Args:
        env: Optional environment mapping. Defaults to `os.environ`.

    Returns:
        Safe publishing configuration.

    Raises:
        ValueError: When publishing is enabled but explicit approvals are missing.
    """

    values = os.environ if env is None else env
    enabled = _env_flag(values, "HUGGINGFACE_PUBLISH", False)
    private = _env_flag(values, "HUGGINGFACE_DATASET_PRIVATE", True)
    allow_public_upload = _env_flag(values, "ALLOW_PUBLIC_DATASET_UPLOAD", False)
    token = values.get("HUGGINGFACE_TOKEN") or None
    repo_id = values.get("HUGGINGFACE_DATASET_REPO") or None

    if enabled and (not token or not repo_id):
        raise ValueError(
            "HUGGINGFACE_PUBLISH=true requires HUGGINGFACE_TOKEN and HUGGINGFACE_DATASET_REPO."
        )
    if enabled and not private and not allow_public_upload:
        raise ValueError(
            "Public dataset upload requires HUGGINGFACE_PUBLISH=true, "
            "HUGGINGFACE_DATASET_PRIVATE=false, and ALLOW_PUBLIC_DATASET_UPLOAD=true."
        )
    return HuggingFacePublishConfig(
        enabled=enabled,
        token=token,
        repo_id=repo_id,
        private=private,
        allow_public_upload=allow_public_upload,
    )


def assert_hf_publish_allowed(paths: Sequence[str | Path], manifest_path: Path | None = None) -> None:
    """Refuse Hugging Face publishing unless every source is public-release approved."""

    if manifest_path is None:
        assert_paths_allowed_for_use(paths, "public_release")
    else:
        assert_paths_allowed_for_use(paths, "public_release", manifest_path)


def assert_openai_training_allowed(paths: Sequence[str | Path], manifest_path: Path | None = None) -> None:
    """Refuse fine-tuning unless every training artifact is training-approved."""

    if manifest_path is None:
        assert_paths_allowed_for_use(paths, "training")
    else:
        assert_paths_allowed_for_use(paths, "training", manifest_path)


class OpenAIFineTuner:
    def __init__(self):
        """Initialize the OpenAI fine-tuner with API key and file paths."""
        # Load environment variables
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully")

        # Define file paths using absolute paths
        self.base_dir = Path(__file__).resolve().parent
        self.train_file = self.base_dir / "OpenAIFineTune" / "stoney_train.jsonl"
        self.valid_file = self.base_dir / "OpenAIFineTune" / "stoney_valid.jsonl"

        # Ensure files exist
        if not self.train_file.exists():
            raise FileNotFoundError(f"Training file not found: {self.train_file}")
        if not self.valid_file.exists():
            raise FileNotFoundError(f"Validation file not found: {self.valid_file}")

        logger.info(f"Found training file: {self.train_file}")
        logger.info(f"Found validation file: {self.valid_file}")
        self.config = load_stoney_config()
        self.fine_tune_model = self.config.openai_finetune_model
        logger.info("Using fine-tune base model: %s", self.fine_tune_model)

        # Hugging Face dataset publishing
        self.hf_publish_config = build_hf_publish_config()
        self.hf_token = self.hf_publish_config.token
        self.hf_repo_id = self.hf_publish_config.repo_id
        self.hf_private = self.hf_publish_config.private
        self.hf_api: Optional[HfApi] = None
        if self.hf_publish_config.enabled:
            assert_hf_publish_allowed([self.train_file, self.valid_file])
            if HfApi is None:
                raise ImportError(
                    "huggingface_hub is required for dataset publishing but is not installed."
                )
            self.hf_api = HfApi(token=self.hf_token)
            logger.info("Hugging Face dataset publishing enabled for repo %s", self.hf_repo_id)
        else:
            logger.warning(
                "Hugging Face dataset publishing disabled. Set HUGGINGFACE_PUBLISH=true, "
                "HUGGINGFACE_TOKEN, and HUGGINGFACE_DATASET_REPO to enable; datasets are "
                "private by default and manifest-gated before upload."
            )

        # Weights & Biases experiment tracking
        self.wandb_api_key = os.getenv("WANDB_API_KEY")
        self.wandb_project = os.getenv("WANDB_PROJECT")
        self.wandb_entity = os.getenv("WANDB_ENTITY") or None
        self.wandb_run_name = os.getenv("WANDB_RUN_NAME")
        self.wandb_enabled = bool(self.wandb_api_key and self.wandb_project)
        self.wandb_run = None

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging if configured."""
        if not self.wandb_enabled:
            return
        if wandb is None:
            raise ImportError("wandb is required for experiment tracking but is not installed.")

        try:
            wandb.login(key=self.wandb_api_key, relogin=True)
            run_name = self.wandb_run_name or f"stoney-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=run_name,
                job_type="openai-fine-tune",
                config={
                    "base_model": self.fine_tune_model,
                    "train_file": str(self.train_file),
                    "valid_file": str(self.valid_file),
                },
            )
            if self.wandb_run:
                logger.info("Weights & Biases run started: %s", self.wandb_run.url)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Unable to initialize Weights & Biases logging: %s", exc)
            self.wandb_run = None
            self.wandb_enabled = False

    def _wandb_log(self, metrics: Dict[str, Any]) -> None:
        """Safely log metrics to Weights & Biases."""
        if not self.wandb_run:
            return
        try:
            wandb.log(metrics)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Failed to log metrics to Weights & Biases: %s", exc)

    def _finish_wandb(self) -> None:
        """Close the active Weights & Biases run."""
        if not self.wandb_run:
            return
        try:
            self.wandb_run.finish()
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Failed to finalize Weights & Biases run: %s", exc)
        finally:
            self.wandb_run = None

    @staticmethod
    def _count_lines(file_path: Path) -> int:
        """Count the number of lines in a JSONL file."""
        with file_path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    def _build_dataset_card(self, train_count: int, valid_count: int, timestamp: str) -> str:
        """Draft a simple dataset card for the Hugging Face repo."""
        return (
            "# Stoney Nakoda Fine-Tuning Dataset\n\n"
            f"- Updated: {timestamp} UTC\n"
            f"- Training examples: {train_count}\n"
            f"- Validation examples: {valid_count}\n\n"
            "This dataset is produced by the automated Stoney Nakoda dictionary-to-fine-tuning "
            "pipeline. It packages conversational training examples used to fine-tune OpenAI "
            "models on Stoney Nakoda language tasks.\n"
        )

    def _publish_dataset_to_hf(self) -> Optional[Dict[str, int]]:
        """Upload the latest training artifacts to Hugging Face Datasets."""
        if not self.hf_api or not self.hf_repo_id:
            return None

        train_count = self._count_lines(self.train_file)
        valid_count = self._count_lines(self.valid_file)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        commit_message = f"Update dataset via automated pipeline ({timestamp} UTC)"

        try:
            self.hf_api.create_repo(
                repo_id=self.hf_repo_id,
                repo_type="dataset",
                private=self.hf_private,
                exist_ok=True,
            )

            uploads = [
                (self.train_file, f"data/{self.train_file.name}"),
                (self.valid_file, f"data/{self.valid_file.name}"),
            ]
            for local_path, remote_path in uploads:
                logger.info(
                    "Uploading %s to Hugging Face dataset repo %s",
                    remote_path,
                    self.hf_repo_id,
                )
                self.hf_api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=remote_path,
                    repo_id=self.hf_repo_id,
                    repo_type="dataset",
                    token=self.hf_token,
                    commit_message=commit_message,
                )

            dataset_card = self._build_dataset_card(train_count, valid_count, timestamp)
            self.hf_api.upload_file(
                path_or_fileobj=io.BytesIO(dataset_card.encode("utf-8")),
                path_in_repo="README.md",
                repo_id=self.hf_repo_id,
                repo_type="dataset",
                token=self.hf_token,
                commit_message=commit_message,
            )

            logger.info("Dataset published to Hugging Face repo %s", self.hf_repo_id)
            return {"train_examples": train_count, "valid_examples": valid_count}
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("Failed to publish dataset to Hugging Face: %s", exc)
            raise

    def upload_file(self, file_path: Path, purpose: str) -> str:
        """Upload a file to OpenAI and return its file ID."""
        file_path = Path(file_path)
        logger.info("Uploading %s file: %s", purpose, file_path)

        with file_path.open("rb") as file:
            response = self.client.files.create(
                file=file,
                purpose=purpose
            )

        logger.info("Successfully uploaded %s file. File ID: %s", purpose, response.id)
        return response.id

    def create_fine_tuning_job(self, training_file_id: str, validation_file_id: str) -> str:
        """Create a fine-tuning job and return its ID."""
        logger.info("Creating fine-tuning job...")
        
        response = self.client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=self.fine_tune_model,
            hyperparameters={
                "n_epochs": 3  # Adjust as needed
            }
        )
        
        logger.info(f"Fine-tuning job created successfully. Job ID: {response.id}")
        return response.id

    def monitor_job_progress(self, job_id: str, check_interval: int = 60):
        """Monitor the progress of a fine-tuning job."""
        logger.info("Starting to monitor fine-tuning job: %s", job_id)
        start_time = time.time()

        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            status = job.status

            # Log detailed status information
            logger.info("Status: %s", status)
            if hasattr(job, 'trained_tokens'):
                logger.info("Trained tokens: %s", job.trained_tokens)
            if hasattr(job, 'training_accuracy'):
                logger.info("Training accuracy: %s", job.training_accuracy)
            if hasattr(job, 'validation_loss'):
                logger.info("Validation loss: %s", job.validation_loss)

            metrics: Dict[str, Any] = {
                "status_index": STATUS_TO_INDEX.get(status, -5),
                "elapsed_seconds": time.time() - start_time,
            }
            if getattr(job, "trained_tokens", None) is not None:
                metrics["trained_tokens"] = job.trained_tokens
            if getattr(job, "training_accuracy", None) is not None:
                metrics["training_accuracy"] = job.training_accuracy
            if getattr(job, "validation_loss", None) is not None:
                metrics["validation_loss"] = job.validation_loss
            self._wandb_log(metrics)

            if status == "succeeded":
                logger.info(" Fine-tuning completed successfully!")
                logger.info("Fine-tuned model ID: %s", job.fine_tuned_model)
                if self.wandb_run:
                    self.wandb_run.summary["final_status"] = status
                    self.wandb_run.summary["fine_tuned_model"] = job.fine_tuned_model
                return job
            elif status == "failed":
                logger.error(" Fine-tuning failed: %s", job.error)
                if self.wandb_run:
                    self.wandb_run.summary["final_status"] = status
                    self.wandb_run.summary["failure_reason"] = str(job.error)
                return job
            elif status in ["cancelled", "expired"]:
                logger.warning(" Fine-tuning job %s", status)
                if self.wandb_run:
                    self.wandb_run.summary["final_status"] = status
                return job

            logger.info("Waiting %d seconds before next check...", check_interval)
            time.sleep(check_interval)

    def run_fine_tuning(self):
        """Run the complete fine-tuning process."""
        dataset_stats: Optional[Dict[str, int]] = None
        try:
            if self.wandb_enabled and not self.wandb_run:
                self._init_wandb()
                self._wandb_log({"stage_index": -1, "stage": "initialization"})

            if self.hf_api:
                logger.info("Publishing dataset to Hugging Face before fine-tuning")
                dataset_stats = self._publish_dataset_to_hf()
                if dataset_stats:
                    if self.wandb_run:
                        self.wandb_run.summary["hf_dataset_repo"] = self.hf_repo_id
                        self.wandb_run.summary["train_examples"] = dataset_stats["train_examples"]
                        self.wandb_run.summary["valid_examples"] = dataset_stats["valid_examples"]
                    self._wandb_log(
                        {
                            "stage_index": 0,
                            "stage": "huggingface_dataset_publish",
                            "train_examples": dataset_stats["train_examples"],
                            "valid_examples": dataset_stats["valid_examples"],
                        }
                    )
            else:
                logger.info("Skipping Hugging Face dataset publishing step (disabled).")

            logger.info("Validating source manifest approval for OpenAI fine-tuning")
            assert_openai_training_allowed([self.train_file, self.valid_file])

            # Step 1: Upload files
            logger.info("Step 1/3: Uploading files to OpenAI")
            self._wandb_log({"stage_index": 1, "stage": "upload_files"})
            train_file_id = self.upload_file(self.train_file, "fine-tune")
            valid_file_id = self.upload_file(self.valid_file, "fine-tune")
            self._wandb_log({"uploaded_train_file": 1, "uploaded_valid_file": 1})
            
            # Step 2: Create fine-tuning job
            logger.info("Step 2/3: Creating fine-tuning job")
            job_id = self.create_fine_tuning_job(train_file_id, valid_file_id)
            if self.wandb_run:
                self.wandb_run.summary["openai_job_id"] = job_id
            self._wandb_log({"stage_index": 2, "stage": "create_job"})
            
            # Step 3: Monitor progress
            logger.info("Step 3/3: Monitoring fine-tuning progress")
            self._wandb_log({"stage_index": 3, "stage": "monitor_progress"})
            final_job = self.monitor_job_progress(job_id)
            if final_job and self.wandb_run:
                result_files = getattr(final_job, "result_files", None)
                if result_files:
                    self.wandb_run.summary["result_files"] = [
                        getattr(file_info, "id", file_info) for file_info in result_files  # type: ignore[arg-type]
                    ]

            return final_job

        except Exception as e:
            logger.error(" Error during fine-tuning process: %s", str(e))
            if self.wandb_run:
                self.wandb_run.summary["final_status"] = "exception"
                self.wandb_run.summary["failure_reason"] = str(e)
            self._wandb_log({"exception": 1})
            raise
        finally:
            self._finish_wandb()

def main():
    """Main function to run the fine-tuning process."""
    logger.info("=== Starting Stoney Language Model Fine-Tuning ===")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        tuner = OpenAIFineTuner()
        tuner.run_fine_tuning()
        
        logger.info("=== Fine-Tuning Process Completed ===")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
