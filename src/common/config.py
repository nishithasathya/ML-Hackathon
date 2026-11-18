"""Project-wide configuration: paths, permitted partitions and file types.

Every module obtains data/output locations from here so that nobody hardcodes
paths. The dataset root can be overridden with the PATHSNAP_DATA_ROOT
environment variable or with a ``--data-root`` command-line argument.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

# src/common/config.py -> parents[2] is the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
# Phase 2: committed, small, reviewable results and the exported inference artefacts.
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
# Cached feature tables (large, git-ignored, re-creatable).
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

# Only these partitions may be read by student-side code. The evaluator-testing
# partition is reserved for the final evaluation (Dataset Exploration, sec. 7)
# and is deliberately NOT listed, so list_trips()/load_trip() refuse it.
TRAIN = "train"
STUDENT_TEST = "student_test"
ALLOWED_PARTITIONS = (TRAIN, STUDENT_TEST)

# The four raw file types supplied for every trip.
FILE_EXTENSIONS = ("track", "nodes", "arcs", "route")

# Fixed seed for any sampling done in exploration/visualisation.
RANDOM_SEED = 42


def get_data_root(override: Optional[PathLike] = None) -> Path:
    """Return the directory that contains the ``train/`` and ``student_test/`` folders."""
    if override:
        return Path(override).expanduser().resolve()
    env_value = os.environ.get("PATHSNAP_DATA_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return DEFAULT_DATA_ROOT


def get_eval_data_root(override: Optional[PathLike] = None) -> Optional[Path]:
    """Directory holding the trips the API serves (``<root>/<trip_id>/<trip_id>.track`` ...).

    This is the ONLY way the evaluator-testing trips enter the system: they are
    supplied by the organisers on Laptop 1 and pointed to with the
    ``PATHSNAP_EVAL_DATA_ROOT`` environment variable. Training and analysis
    code never reads this directory (it goes through ``check_partition``).
    Returns None when nothing is configured, so callers can fail clearly.
    """
    value = override or os.environ.get("PATHSNAP_EVAL_DATA_ROOT")
    return Path(value).expanduser().resolve() if value else None


def get_results_dir(*parts: str, create: bool = True) -> Path:
    """Return ``results/<parts...>`` (committed experiment evidence)."""
    path = Path(os.environ.get("PATHSNAP_RESULTS_ROOT", DEFAULT_RESULTS_ROOT)).joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_artifact_dir(override: Optional[PathLike] = None) -> Path:
    """Directory with the exported model bundle (``PATHSNAP_ARTIFACT_DIR`` overrides)."""
    value = override or os.environ.get("PATHSNAP_ARTIFACT_DIR")
    return Path(value).expanduser().resolve() if value else DEFAULT_ARTIFACT_ROOT


def get_processed_dir(*parts: str, create: bool = True) -> Path:
    path = Path(os.environ.get("PATHSNAP_PROCESSED_ROOT", DEFAULT_PROCESSED_ROOT)).joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_dir(*parts: str, root: Optional[PathLike] = None, create: bool = True) -> Path:
    """Return ``outputs/<parts...>`` (or ``<root>/<parts...>``), creating it if requested."""
    base = Path(root).expanduser().resolve() if root else DEFAULT_OUTPUT_ROOT
    path = base.joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def check_partition(partition: str) -> str:
    """Raise ValueError unless ``partition`` is one students are allowed to use."""
    if partition not in ALLOWED_PARTITIONS:
        raise ValueError(
            f"Partition '{partition}' is not permitted. Allowed partitions: "
            f"{', '.join(ALLOWED_PARTITIONS)}. Evaluator-testing data must not be used."
        )
    return partition
