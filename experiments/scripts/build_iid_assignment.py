from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.fl.iid_assignment import (
    load_assignment_array,
    partitions_to_assignment_array,
    save_assignment_array,
)
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fl"
    / "iid"
)

ASSIGNMENT_PATH = (
    OUTPUT_DIR
    / "train_client_assignments.npy"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "train_client_assignments_summary.json"
)


# ----------------------------------------------------------------------
# Load real training metadata
# ----------------------------------------------------------------------


def load_training_metadata(
    loader: NBaIoTSplitLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load global training-row ordinals and binary labels.

    The feature columns are never retained in memory.

    The row ordinal is defined by the exact order produced by:

        loader.iter_chunks("train")

    Therefore assignment index i corresponds to the i-th training
    row in the frozen source-index order.
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
        raise RuntimeError(
            "No N-BaIoT training rows were loaded."
        )

    return (
        np.concatenate(row_ids),
        np.concatenate(labels),
    )


# ----------------------------------------------------------------------
# Build assignment
# ----------------------------------------------------------------------


def build_assignment(
    row_indices: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Recreate the validated stratified IID partition and convert it
    into a compact assignment array.
    """

    partitions = make_stratified_iid_partitions(
        row_indices=row_indices,
        binary_labels=labels,
        num_clients=NUM_CLIENTS,
        seed=SEED,
    )

    # Validate the partition before converting it.
    validate_partitions(
        partitions,
        row_indices,
    )

    assignment = partitions_to_assignment_array(
        partitions=partitions,
        total_rows=len(row_indices),
    )

    return assignment


# ----------------------------------------------------------------------
# Audit assignment
# ----------------------------------------------------------------------


def audit_assignment(
    assignment: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Verify the real assignment and generate a reproducible summary.
    """

    if len(assignment) != len(labels):
        raise RuntimeError(
            "Assignment and label arrays have different lengths."
        )

    unique_clients = np.unique(assignment)

    expected_clients = np.arange(
        NUM_CLIENTS,
        dtype=np.int8,
    )

    if not np.array_equal(
        unique_clients,
        expected_clients,
    ):
        raise RuntimeError(
            "Assignment does not contain exactly the expected "
            f"client indices 0..{NUM_CLIENTS - 1}.\n"
            f"Found: {unique_clients.tolist()}"
        )

    client_summaries = []

    global_total = len(labels)
    global_benign = int(np.sum(labels == 0))
    global_attack = int(np.sum(labels == 1))

    global_benign_pct = (
        global_benign / global_total * 100
    )

    global_attack_pct = (
        global_attack / global_total * 100
    )

    print()
    print("=" * 80)
    print("REAL IID ASSIGNMENT DISTRIBUTION")
    print("=" * 80)

    print()
    print(
        f"Global training rows: {global_total:,}"
    )

    print(
        f"Global benign: {global_benign:,} "
        f"({global_benign_pct:.6f}%)"
    )

    print(
        f"Global attack: {global_attack:,} "
        f"({global_attack_pct:.6f}%)"
    )

    print()

    for client_index in range(NUM_CLIENTS):
        client_mask = assignment == client_index
        client_labels = labels[client_mask]

        total = len(client_labels)
        benign = int(np.sum(client_labels == 0))
        attack = int(np.sum(client_labels == 1))

        benign_pct = (
            benign / total * 100
        )

        attack_pct = (
            attack / total * 100
        )

        client_summaries.append(
            {
                "client_index": client_index,
                "client_id": f"client_{client_index + 1}",
                "rows": total,
                "benign_rows": benign,
                "attack_rows": attack,
                "benign_percentage": benign_pct,
                "attack_percentage": attack_pct,
            }
        )

        print(
            f"client_{client_index + 1}: "
            f"rows={total:,}, "
            f"benign={benign:,} ({benign_pct:.6f}%), "
            f"attack={attack:,} ({attack_pct:.6f}%)"
        )

    client_row_counts = [
        item["rows"]
        for item in client_summaries
    ]

    client_benign_percentages = [
        item["benign_percentage"]
        for item in client_summaries
    ]

    client_attack_percentages = [
        item["attack_percentage"]
        for item in client_summaries
    ]

    max_benign_deviation = max(
        abs(
            percentage - global_benign_pct
        )
        for percentage in client_benign_percentages
    )

    max_attack_deviation = max(
        abs(
            percentage - global_attack_pct
        )
        for percentage in client_attack_percentages
    )

    print()
    print("=" * 80)
    print("REAL IID ASSIGNMENT VALIDATION")
    print("=" * 80)

    print()
    print(
        f"Minimum client rows: "
        f"{min(client_row_counts):,}"
    )

    print(
        f"Maximum client rows: "
        f"{max(client_row_counts):,}"
    )

    print(
        f"Maximum benign-ratio deviation: "
        f"{max_benign_deviation:.6f} percentage points"
    )

    print(
        f"Maximum attack-ratio deviation: "
        f"{max_attack_deviation:.6f} percentage points"
    )

    if sum(client_row_counts) != global_total:
        raise RuntimeError(
            "Client assignment counts do not sum to "
            "the complete training dataset."
        )

    if max_benign_deviation > 0.01:
        raise RuntimeError(
            "IID benign-ratio deviation exceeds "
            "0.01 percentage points."
        )

    if max_attack_deviation > 0.01:
        raise RuntimeError(
            "IID attack-ratio deviation exceeds "
            "0.01 percentage points."
        )

    print()
    print("Assignment length check: PASS")
    print("Client index check: PASS")
    print("Complete row coverage check: PASS")
    print("Class-ratio preservation check: PASS")

    return {
        "num_clients": NUM_CLIENTS,
        "seed": SEED,
        "total_training_rows": global_total,
        "global_benign_rows": global_benign,
        "global_attack_rows": global_attack,
        "global_benign_percentage": global_benign_pct,
        "global_attack_percentage": global_attack_pct,
        "minimum_client_rows": min(client_row_counts),
        "maximum_client_rows": max(client_row_counts),
        "maximum_benign_ratio_deviation_percentage_points": (
            max_benign_deviation
        ),
        "maximum_attack_ratio_deviation_percentage_points": (
            max_attack_deviation
        ),
        "clients": client_summaries,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    print("=" * 80)
    print("BUILD REAL N-BAIoT STRATIFIED IID ASSIGNMENT")
    print("=" * 80)

    loader = NBaIoTSplitLoader(
        project_root=PROJECT_ROOT,
        chunk_size=50_000,
    )

    print()
    print("Validating source split indexes...")

    loader.validate_source_indexes()

    print("Source split indexes: PASS")

    print()
    print("Loading real N-BaIoT training metadata...")
    print("Feature matrices are NOT retained in memory.")

    row_indices, labels = load_training_metadata(
        loader
    )

    print(
        f"Training rows loaded: "
        f"{len(row_indices):,}"
    )

    print(
        f"Training labels loaded: "
        f"{len(labels):,}"
    )

    if len(row_indices) != EXPECTED_TRAINING_ROWS:
        raise RuntimeError(
            "Unexpected training row count.\n"
            f"Expected: {EXPECTED_TRAINING_ROWS:,}\n"
            f"Actual:   {len(row_indices):,}"
        )

    if len(labels) != EXPECTED_TRAINING_ROWS:
        raise RuntimeError(
            "Unexpected training label count.\n"
            f"Expected: {EXPECTED_TRAINING_ROWS:,}\n"
            f"Actual:   {len(labels):,}"
        )

    print("Training row-count check: PASS")

    unique_labels = np.unique(labels)

    if not np.array_equal(
        unique_labels,
        np.array([0, 1]),
    ):
        raise RuntimeError(
            "Expected exactly binary labels 0 and 1.\n"
            f"Found: {unique_labels.tolist()}"
        )

    print("Binary-label check: PASS")

    print()
    print("Creating deterministic stratified IID partitions...")

    assignment = build_assignment(
        row_indices=row_indices,
        labels=labels,
    )

    print("Partition creation: PASS")

    print()
    print("Saving IID assignment artifact...")

    save_assignment_array(
        assignment=assignment,
        path=ASSIGNMENT_PATH,
    )

    print(
        f"Assignment saved:\n"
        f"{ASSIGNMENT_PATH}"
    )

    print()
    print("Reloading saved assignment for verification...")

    loaded_assignment = load_assignment_array(
        path=ASSIGNMENT_PATH,
        expected_rows=EXPECTED_TRAINING_ROWS,
        expected_clients=NUM_CLIENTS,
    )

    np.testing.assert_array_equal(
        loaded_assignment,
        assignment,
    )

    print("Saved assignment verification: PASS")

    summary = audit_assignment(
        assignment=loaded_assignment,
        labels=labels,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary["assignment_file"] = str(
        ASSIGNMENT_PATH.relative_to(PROJECT_ROOT)
    )

    summary["assignment_definition"] = (
        "Global sequential training-row ordinal generated "
        "by loader.iter_chunks('train')."
    )

    summary["partition_type"] = (
        "stratified_iid"
    )

    summary["client_index_definition"] = (
        "0-based client index corresponding to client_1 "
        "through client_9."
    )

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )
        handle.write("\n")

    print()
    print(
        f"Summary saved:\n"
        f"{SUMMARY_PATH}"
    )

    print()
    print("=" * 80)
    print("REAL IID ASSIGNMENT BUILD: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()