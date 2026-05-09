"""Print safe diagnostics for Stoney Nakoda pipeline configuration."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stoney_config import ConfigError, load_stoney_config


def main() -> int:
    """Validate and print purpose-specific model configuration."""

    try:
        config = load_stoney_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    print(config.diagnostics())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
