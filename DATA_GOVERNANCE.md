# Data Governance

This repository contains Stoney Nakoda language materials and generated derivatives. Code can enforce workflow gates, schema checks, and safe defaults, but it cannot certify cultural permission, speaker approval, rights ownership, or community authority. Those decisions must remain with the appropriate human and community review process.

## Principles

The repository follows a default-deny model for language data. A source must be explicitly reviewed in `SOURCE_MANIFEST.yml` before it can be used for training, publishing, or public release.

Governance decisions should be guided by Indigenous data governance principles, including ownership, control, access, possession, collective benefit, authority to control, responsibility, and ethics. When the repository state is uncertain, code and documentation must say so plainly.

## Required Source Manifest

Every source artifact category must have a record in `SOURCE_MANIFEST.yml`. Each record includes:

- `source_id`
- `path_glob`
- `data_type`
- `origin`
- `rights_holder`
- `permission_status`
- `license`
- `allowed_uses`
- `prohibited_uses`
- `contains_sensitive_content`
- `review_contact`
- `notes`

Uncertain records must use `permission_status: unknown`. Do not infer approval from a file being present in the repository.

## Permission Statuses

- `unknown`: No verified governance decision is recorded.
- `internal_only`: Material may be inspected locally by authorized project participants only.
- `approved_for_training`: Material may be used for model training workflows, subject to `allowed_uses`.
- `approved_for_public_release`: Material may be uploaded, published, or released publicly, subject to `allowed_uses`.
- `restricted`: Material must not be used for training, upload, publishing, or public release.
- `removed_pending_review`: Material should be removed from active workflows until reviewed.

## Workflow Gates

Training workflows must require `permission_status: approved_for_training` for every source file that enters training.

Upload, publishing, and public-release workflows must require `permission_status: approved_for_public_release` for every source file being uploaded or published.

Unknown, internal-only, restricted, and removed-pending-review sources must be refused by default.

## Validation

Run the manifest validator before any training or publishing workflow:

```bash
python scripts/validate_source_manifest.py
```

Use the policy check when adding workflow code:

```python
from scripts.validate_source_manifest import assert_paths_allowed_for_use

assert_paths_allowed_for_use(["OpenAIFineTune/stoney_train.jsonl"], "training")
assert_paths_allowed_for_use(["OpenAIFineTune/stoney_train.jsonl"], "public_release")
```

## Removal and Corrections

If a rights holder, community reviewer, or source steward asks for removal or restricted handling, update `SOURCE_MANIFEST.yml` immediately to `removed_pending_review` or `restricted`, stop downstream workflows, and document the decision in the manifest notes. Do not delete or rewrite tracked data automatically; removal and redaction are governance decisions.
