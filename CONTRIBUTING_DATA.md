# Contributing Data

Data contributions must be reviewed before they enter training, evaluation, publishing, or release workflows.

## Add a Source

1. Place the file in the appropriate repository location only if it is allowed to be stored there.
2. Add or update a record in `SOURCE_MANIFEST.yml`.
3. Set `permission_status: unknown` unless a documented governance decision already exists.
4. Leave `allowed_uses` empty until review approves a specific use.
5. Add clear `prohibited_uses` for training, upload, publishing, and public release while the source is uncertain.
6. Run `python scripts/validate_source_manifest.py`.

## Required Review

Before a source is used for training, its manifest record must be changed by a human reviewer to `permission_status: approved_for_training` and include `training` in `allowed_uses`.

Before a source is uploaded, published, or publicly released, its manifest record must be changed by a human reviewer to `permission_status: approved_for_public_release` and include `public_release` in `allowed_uses`.

The code must not mark data as speaker-reviewed, community-approved, training-approved, or public-release-approved automatically.

## Generated Data

Generated Q&A, fine-tuning JSONL, grammar extractions, curated grammar rules, and RL tasks are derivative artifacts. They still require manifest records and must inherit the most restrictive governance status of their sources until reviewed.

Do not commit large generated artifacts unless a maintainer explicitly asks for them and the manifest allows the intended use.

## Acceptance Check

For any PR that changes data handling:

```bash
python scripts/validate_source_manifest.py
pytest
```

The PR description should name the manifest records it changed and the acceptance check that was run.
