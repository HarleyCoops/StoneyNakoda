from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.validate_source_manifest import (
    ManifestPolicyError,
    assert_paths_allowed_for_use,
    validate_manifest,
)


def _record(status: str, allowed_uses: list[str] | None = None, prohibited_uses: list[str] | None = None) -> dict:
    return {
        "source_id": "fixture_source",
        "path_glob": "OpenAIFineTune/*.jsonl",
        "data_type": "fine_tuning_dataset",
        "origin": "fixture",
        "rights_holder": "fixture",
        "permission_status": status,
        "license": "fixture",
        "allowed_uses": allowed_uses or [],
        "prohibited_uses": prohibited_uses or [],
        "contains_sensitive_content": True,
        "review_contact": "fixture reviewer",
        "notes": "fixture",
    }


def _write_manifest(tmp_path: Path, records: list[dict]) -> Path:
    manifest_path = tmp_path / "SOURCE_MANIFEST.yml"
    manifest_path.write_text(
        yaml.safe_dump({"schema_version": "1.0", "records": records}, sort_keys=False),
        encoding="utf-8",
    )
    return manifest_path


def test_repository_source_manifest_validates() -> None:
    validate_manifest()


@pytest.mark.parametrize("status", ["unknown", "internal_only", "restricted", "removed_pending_review"])
def test_unapproved_sources_cannot_enter_training(status: str, tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [_record(status)])

    with pytest.raises(ManifestPolicyError, match="refuses training"):
        assert_paths_allowed_for_use(["OpenAIFineTune/stoney_train.jsonl"], "training", manifest_path)


def test_training_requires_training_approval_and_allowed_use(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [_record("approved_for_training", allowed_uses=["training"])],
    )

    assert_paths_allowed_for_use(["OpenAIFineTune/stoney_train.jsonl"], "training", manifest_path)


def test_training_approval_does_not_allow_public_upload(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [_record("approved_for_training", allowed_uses=["training"])],
    )

    with pytest.raises(ManifestPolicyError, match="refuses public_release"):
        assert_paths_allowed_for_use(["OpenAIFineTune/stoney_train.jsonl"], "public_release", manifest_path)


def test_public_upload_requires_public_release_approval(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [_record("approved_for_public_release", allowed_uses=["public_release"])],
    )

    assert_paths_allowed_for_use(["OpenAIFineTune/stoney_train.jsonl"], "public_release", manifest_path)


def test_missing_manifest_record_refuses_workflow(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [_record("approved_for_training", allowed_uses=["training"])])

    with pytest.raises(ManifestPolicyError, match="no SOURCE_MANIFEST.yml record"):
        assert_paths_allowed_for_use(["data/training_datasets_stoney.jsonl"], "training", manifest_path)
