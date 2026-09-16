from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Project root / import path
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------------------------------
# Project imports
# ----------------------------------------------------------------------

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.fl.partitioning import (
    make_stratified_iid_partitions,
    validate_partitions,
)


# ----------------------------------------------------------------------
# Experiment constants
# ----------------------------------------------------------------------

NUM_CLIENTS = 9
SEED = 42

EXPECTED_TRAINING_ROWS = 4_943_824


# ----------------------------------------------------------------------
# Load training row metadata
# ----------------------------------------------------------------------

def load_training_row_metadata(
    loader: NBaIoTSplitLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load only training-row identifiers and binary labels.

    The 115 feature columns are not retained in memory.

    The generated row IDs are experiment-level sequential identifiers.
    They are used only to validate partition assignment in this audit.
    They are not source-file row numbers.
    """

    row_ids = []
    labels = []

    global_row_id = 0

    for chunk in loader.iter_chunks("train"):
        chunk_size = len(chunk.binary_labels)

        chunk_row_ids = np.arange(
            global_row_id,
            global_row_id + chunk_size,
            dtype=np.int64,
        )

        chunk_labels = chunk.binary_labels.to_numpy(
            dtype=np.int64
        )

        row_ids.append(chunk_row_ids)
        labels.append(chunk_labels)

        global_row_id += chunk_size

    if not row_ids:
        raise RuntimeError("No training rows were loaded.")

    return (
        np.concatenate(row_ids),
        np.concatenate(labels),
    )


# ----------------------------------------------------------------------
# Main audit
# ----------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("REAL N-BaIoT STRATIFIED IID PARTITION AUDIT")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Create loader
    # ------------------------------------------------------------------

    loader = NBaIoTSplitLoader(
        project_root=PROJECT_ROOT,
        chunk_size=50_000,
    )

    # ------------------------------------------------------------------
    # Validate frozen source split indexes
    # ------------------------------------------------------------------

    print()
    print("Validating source split indexes...")

    loader.validate_source_indexes()

    print("Source split indexes: PASS")

    # ------------------------------------------------------------------
    # Load training metadata
    # ------------------------------------------------------------------

    print()
    print("Loading training row metadata...")
    print("Features are NOT loaded into memory.")

    row_indices, labels = load_training_row_metadata(loader)

    print(f"Training rows loaded: {len(row_indices):,}")
    print(f"Label metadata rows:   {len(labels):,}")

    # ------------------------------------------------------------------
    # Validate training row count
    # ------------------------------------------------------------------

    if len(row_indices) != EXPECTED_TRAINING_ROWS:
        raise RuntimeError(
            "Unexpected training row count: "
            f"{len(row_indices):,}; "
            f"expected {EXPECTED_TRAINING_ROWS:,}."
        )

    if len(labels) != EXPECTED_TRAINING_ROWS:
        raise RuntimeError(
            "Unexpected training label count: "
            f"{len(labels):,}; "
            f"expected {EXPECTED_TRAINING_ROWS:,}."
        )

    print("Training row-count check: PASS")

    # ------------------------------------------------------------------
    # Validate binary labels
    # ------------------------------------------------------------------

    unique_labels = np.unique(labels)

    if not np.array_equal(unique_labels, np.array([0, 1])):
        raise RuntimeError(
            "Training labels must contain exactly binary labels "
            f"0 and 1. Found: {unique_labels.tolist()}"
        )

    print("Binary-label check: PASS")

    # ------------------------------------------------------------------
    # Create stratified IID partitions
    # ------------------------------------------------------------------

    print()
    print("Creating stratified IID partitions...")

    partitions = make_stratified_iid_partitions(
        row_indices=row_indices,
        binary_labels=labels,
        num_clients=NUM_CLIENTS,
        seed=SEED,
    )

    print("Partition creation: PASS")

    # ------------------------------------------------------------------
    # Validate complete row assignment
    # ------------------------------------------------------------------

    print()
    print("Validating complete row assignment...")

    validate_partitions(
        partitions,
        row_indices,
    )

    print("Partition coverage: PASS")

    # ------------------------------------------------------------------
    # Global distribution
    # ------------------------------------------------------------------

    global_total = len(labels)

    global_benign = int(np.sum(labels == 0))
    global_attack = int(np.sum(labels == 1))

    global_benign_pct = (
        global_benign / global_total * 100
    )

    global_attack_pct = (
        global_attack / global_total * 100
    )

    # ------------------------------------------------------------------
    # Print client distributions
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("CLIENT DISTRIBUTIONS")
    print("=" * 80)

    print()
    print(
        f"Global training distribution: "
        f"benign={global_benign:,} "
        f"({global_benign_pct:.4f}%), "
        f"attack={global_attack:,} "
        f"({global_attack_pct:.4f}%)"
    )

    client_benign_percentages = []
    client_attack_percentages = []
    client_row_counts = []

    for partition in partitions:
        client_labels = labels[partition.row_indices]

        total = len(client_labels)

        benign = int(np.sum(client_labels == 0))
        attack = int(np.sum(client_labels == 1))

        benign_pct = benign / total * 100
        attack_pct = attack / total * 100

        client_row_counts.append(total)
        client_benign_percentages.append(benign_pct)
        client_attack_percentages.append(attack_pct)

        print()
        print(partition.client_id)
        print("-" * len(partition.client_id))
        print(f"Rows:   {total:,}")
        print(f"Benign: {benign:,} ({benign_pct:.4f}%)")
        print(f"Attack: {attack:,} ({attack_pct:.4f}%)")

    # ------------------------------------------------------------------
    # Additional IID balance checks
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("PARTITION BALANCE CHECKS")
    print("=" * 80)

    min_rows = min(client_row_counts)
    max_rows = max(client_row_counts)

    print()
    print(f"Minimum client rows: {min_rows:,}")
    print(f"Maximum client rows: {max_rows:,}")

    expected_rows_per_client = (
        global_total / NUM_CLIENTS
    )

    print(
        f"Expected average rows/client: "
        f"{expected_rows_per_client:,.2f}"
    )

    max_benign_deviation = max(
        abs(pct - global_benign_pct)
        for pct in client_benign_percentages
    )

    max_attack_deviation = max(
        abs(pct - global_attack_pct)
        for pct in client_attack_percentages
    )

    print(
        f"Maximum benign-ratio deviation: "
        f"{max_benign_deviation:.6f} percentage points"
    )

    print(
        f"Maximum attack-ratio deviation: "
        f"{max_attack_deviation:.6f} percentage points"
    )

    # ------------------------------------------------------------------
    # Final checks
    # ------------------------------------------------------------------

    if sum(client_row_counts) != global_total:
        raise RuntimeError(
            "Client row counts do not sum to the global training row count."
        )

    if max_benign_deviation > 0.01:
        raise RuntimeError(
            "Stratified IID benign-class ratio deviates by more "
            "than 0.01 percentage points from the global ratio."
        )

    if max_attack_deviation > 0.01:
        raise RuntimeError(
            "Stratified IID attack-class ratio deviates by more "
            "than 0.01 percentage points from the global ratio."
        )

    print()
    print("Row-count balance check: PASS")
    print("Class-ratio preservation check: PASS")

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("IID PARTITION AUDIT: PASS")
    print("=" * 80)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()