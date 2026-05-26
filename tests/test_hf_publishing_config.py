from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from openai_finetune import (
    assert_hf_publish_allowed,
    assert_openai_training_allowed,
    build_hf_publish_config,
)
from scripts.validate_source_manifest import ManifestPolicyError


def _record(status: str, allowed_uses: list[str] | None = None) -> dict:
    return {
        "source_id": "fixture_finetune_exports",
        "path_glob": "OpenAIFineTune/*.jsonl",
        "data_type": "fine_tuning_dataset",
        "origin": "fixture",
        "rights_holder": "fixture",
        "permission_status": status,
        "license": "fixture",
        "allowed_uses": allowed_uses or [],
        "prohibited_uses": [],
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


def test_huggingface_publishing_disabled_by_default() -> None:
    config = build_hf_publish_config({})

    assert config.enabled is False
    assert config.private is True


def test_huggingface_token_does_not_enable_upload_without_publish_flag() -> None:
    config = build_hf_publish_config(
        {
            "HUGGINGFACE_TOKEN": "hf_token",
            "HUGGINGFACE_DATASET_REPO": "owner/repo",
        }
    )

    assert config.enabled is False


def test_public_upload_requires_second_explicit_flag() -> None:
    with pytest.raises(ValueError, match="ALLOW_PUBLIC_DATASET_UPLOAD=true"):
        build_hf_publish_config(
            {
                "HUGGINGFACE_PUBLISH": "true",
                "HUGGINGFACE_DATASET_PRIVATE": "false",
                "HUGGINGFACE_TOKEN": "hf_token",
                "HUGGINGFACE_DATASET_REPO": "owner/repo",
            }
        )


def test_private_publish_config_requires_token_and_repo() -> None:
    config = build_hf_publish_config(
        {
            "HUGGINGFACE_PUBLISH": "true",
            "HUGGINGFACE_DATASET_PRIVATE": "true",
            "HUGGINGFACE_TOKEN": "hf_token",
            "HUGGINGFACE_DATASET_REPO": "owner/repo",
        }
    )

    assert config.enabled is True
    assert config.private is True


def test_manifest_blocks_unknown_sources_before_hf_upload(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [_record("unknown")])

    with pytest.raises(ManifestPolicyError, match="refuses public_release"):
        assert_hf_publish_allowed(
            ["OpenAIFineTune/stoney_train.jsonl", "OpenAIFineTune/stoney_valid.jsonl"],
            manifest_path,
        )


def test_manifest_allows_public_release_sources_for_hf_upload(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [_record("approved_for_public_release", allowed_uses=["public_release"])],
    )

    assert_hf_publish_allowed(
        ["OpenAIFineTune/stoney_train.jsonl", "OpenAIFineTune/stoney_valid.jsonl"],
        manifest_path,
    )


def test_manifest_blocks_unknown_sources_before_openai_training(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [_record("unknown")])

    with pytest.raises(ManifestPolicyError, match="refuses training"):
        assert_openai_training_allowed(
            ["OpenAIFineTune/stoney_train.jsonl", "OpenAIFineTune/stoney_valid.jsonl"],
            manifest_path,
        )


def test_manifest_allows_training_approved_sources_for_openai_training(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [_record("approved_for_training", allowed_uses=["training"])],
    )

    assert_openai_training_allowed(
        ["OpenAIFineTune/stoney_train.jsonl", "OpenAIFineTune/stoney_valid.jsonl"],
        manifest_path,
    )
