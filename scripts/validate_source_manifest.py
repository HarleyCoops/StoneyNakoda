"""Validate source governance manifests and enforce use-specific gates."""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import jsonschema
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "SOURCE_MANIFEST.yml"
DEFAULT_SCHEMA_PATH = REPO_ROOT / "schemas" / "source_manifest.schema.json"

PERMISSION_STATUSES = {
    "unknown",
    "internal_only",
    "approved_for_training",
    "approved_for_public_release",
    "restricted",
    "removed_pending_review",
}

REQUIRED_STATUS_BY_USE = {
    "training": "approved_for_training",
    "public_release": "approved_for_public_release",
    "upload": "approved_for_public_release",
    "publishing": "approved_for_public_release",
}

ALLOWED_USE_ALIASES = {
    "training": {"training"},
    "public_release": {"public_release", "upload", "publishing"},
    "upload": {"public_release", "upload", "publishing"},
    "publishing": {"public_release", "upload", "publishing"},
}


class ManifestPolicyError(ValueError):
    """Raised when a source manifest does not allow a requested workflow use."""


def load_yaml(path: Path) -> Any:
    """Load a YAML file from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Load a source manifest file."""

    payload = load_yaml(path)
    if not isinstance(payload, dict):
        raise ManifestPolicyError(f"Manifest must be an object: {path}")
    return payload


def load_schema(path: Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    """Load a JSON schema file."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ManifestPolicyError(f"Schema must be an object: {path}")
    return payload


def validate_manifest_dict(
    manifest: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
) -> None:
    """Validate a source manifest object against the repository schema."""

    jsonschema.Draft202012Validator.check_schema(schema or load_schema())
    validator = jsonschema.Draft202012Validator(schema or load_schema())
    errors = sorted(validator.iter_errors(manifest), key=lambda err: list(err.path))
    if errors:
        details = "\n".join(f"- {'/'.join(map(str, err.path)) or '<root>'}: {err.message}" for err in errors)
        raise ManifestPolicyError(f"Source manifest validation failed:\n{details}")


def validate_manifest(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> dict[str, Any]:
    """Load and validate a manifest file, returning the parsed object."""

    manifest = load_manifest(manifest_path)
    schema = load_schema(schema_path)
    validate_manifest_dict(manifest, schema)
    return manifest


def normalize_repo_path(path: str | Path, repo_root: Path = REPO_ROOT) -> str:
    """Return a stable POSIX-style path relative to the repository root."""

    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ManifestPolicyError(f"Path is outside repository root: {path}") from exc
    return candidate.as_posix()


def find_records_for_path(path: str | Path, manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return all manifest records whose glob covers the supplied path."""

    normalized = normalize_repo_path(path)
    records = manifest.get("records", [])
    if not isinstance(records, list):
        raise ManifestPolicyError("Manifest records must be a list")
    return [
        record
        for record in records
        if isinstance(record, Mapping)
        and fnmatch.fnmatchcase(normalized, str(record.get("path_glob", "")))
    ]


def _record_allows_use(record: Mapping[str, Any], intended_use: str) -> bool:
    required_status = REQUIRED_STATUS_BY_USE[intended_use]
    allowed_aliases = ALLOWED_USE_ALIASES[intended_use]
    status = record.get("permission_status")
    allowed_uses = set(record.get("allowed_uses") or [])
    prohibited_uses = set(record.get("prohibited_uses") or [])
    return (
        status == required_status
        and bool(allowed_uses & allowed_aliases)
        and not bool(prohibited_uses & allowed_aliases)
    )


def assert_paths_allowed_for_use(
    paths: Sequence[str | Path],
    intended_use: str,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> None:
    """Raise unless every supplied path is approved for the requested workflow use.

    Args:
        paths: Source or derived artifact paths to check.
        intended_use: One of training, public_release, upload, or publishing.
        manifest_path: Source manifest path.

    Raises:
        ManifestPolicyError: When any path is missing from the manifest or not approved.
    """

    if intended_use not in REQUIRED_STATUS_BY_USE:
        allowed = ", ".join(sorted(REQUIRED_STATUS_BY_USE))
        raise ManifestPolicyError(f"Unsupported intended use '{intended_use}'. Expected one of: {allowed}")

    manifest = validate_manifest(manifest_path)
    failures: list[str] = []
    required_status = REQUIRED_STATUS_BY_USE[intended_use]
    for path in paths:
        normalized = normalize_repo_path(path)
        records = find_records_for_path(normalized, manifest)
        if not records:
            failures.append(f"{normalized}: no SOURCE_MANIFEST.yml record covers this path")
            continue
        if not any(_record_allows_use(record, intended_use) for record in records):
            statuses = ", ".join(
                f"{record.get('source_id')}={record.get('permission_status')}" for record in records
            )
            failures.append(
                f"{normalized}: requires {required_status} with allowed use '{intended_use}'; found {statuses}"
            )

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise ManifestPolicyError(f"Source manifest refuses {intended_use} workflow:\n{details}")


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SOURCE_MANIFEST.yml and optional workflow gates.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--check-use", choices=sorted(REQUIRED_STATUS_BY_USE))
    parser.add_argument("--paths", nargs="*", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run source manifest validation from the command line."""

    args = _parse_args(argv or sys.argv[1:])
    try:
        validate_manifest(args.manifest, args.schema)
        if args.check_use:
            if not args.paths:
                raise ManifestPolicyError("--paths is required when --check-use is supplied")
            assert_paths_allowed_for_use(args.paths, args.check_use, args.manifest)
    except ManifestPolicyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Source manifest valid: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
