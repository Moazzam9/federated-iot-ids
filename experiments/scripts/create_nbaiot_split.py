from __future__ import annotations

import csv
import hashlib
import sqlite3
from collections import defaultdict
from pathlib import Path
from random import Random


PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIT_DB = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "inspection"
    / "dataset_duplicate_audit.sqlite"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

GROUP_ASSIGNMENTS_DB = (
    OUTPUT_DIR
    / "nbaiot_group_assignments.sqlite"
)

DUPLICATE_ASSIGNMENTS_CSV = (
    OUTPUT_DIR
    / "nbaiot_duplicate_group_assignments.csv"
)

GROUP_SUMMARY_CSV = (
    OUTPUT_DIR
    / "nbaiot_split_group_summary.csv"
)

SPLIT_SPECIFICATION_TXT = (
    OUTPUT_DIR
    / "split_specification.txt"
)

RANDOM_SEED = 42

SPLIT_NAMES = (
    "train",
    "validation",
    "test",
)

SPLIT_TARGETS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

EXPECTED_TOTAL_ROWS = 7_062_606


def normalize_label(value) -> int:
    label = str(value).strip().lower()

    if label == "benign":
        return 0

    if label in ("gafgyt", "mirai", "attack"):
        return 1

    raise RuntimeError(
        f"Unexpected label value in audit database: {value!r}"
    )


def label_name(label: int) -> str:
    if label == 0:
        return "benign"

    if label == 1:
        return "attack"

    raise RuntimeError(
        f"Unexpected binary label: {label}"
    )


def make_deterministic_seed(
    device: str,
    binary_label: int,
) -> int:

    seed_text = (
        f"{RANDOM_SEED}|"
        f"{device}|"
        f"{binary_label}"
    )

    digest = hashlib.sha256(
        seed_text.encode("utf-8")
    ).hexdigest()

    return int(digest[:16], 16)


def assign_groups_to_partitions(
    groups: list[tuple[str, int]],
    seed: int,
) -> dict[str, str]:

    if not groups:
        return {}

    rng = Random(seed)

    groups = list(groups)

    rng.shuffle(groups)

    groups.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    total_rows = sum(
        count
        for _, count in groups
    )

    targets = {
        split: total_rows * SPLIT_TARGETS[split]
        for split in SPLIT_NAMES
    }

    current = {
        split: 0
        for split in SPLIT_NAMES
    }

    assignments = {}

    for feature_hash, count in groups:

        best_split = None
        best_score = None

        for split in SPLIT_NAMES:

            candidate_counts = current.copy()

            candidate_counts[split] += count

            score = sum(
                abs(
                    candidate_counts[name]
                    - targets[name]
                )
                for name in SPLIT_NAMES
            )

            tie_break = candidate_counts[split]

            candidate = (
                score,
                tie_break,
            )

            if (
                best_score is None
                or candidate < best_score
            ):
                best_score = candidate
                best_split = split

        if best_split is None:
            raise RuntimeError(
                "Unable to assign feature-vector group."
            )

        assignments[feature_hash] = best_split

        current[best_split] += count

    return assignments


def main() -> None:

    print("=" * 70)
    print("N-BaIoT Leakage-Safe Dataset Splitter")
    print("=" * 70)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Audit database: {AUDIT_DB}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    if not AUDIT_DB.exists():
        raise FileNotFoundError(
            "Required audit database was not found:\n"
            f"{AUDIT_DB}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_connection = sqlite3.connect(
        AUDIT_DB
    )

    source_connection.row_factory = sqlite3.Row

    source_cursor = source_connection.cursor()

    try:

        print("Checking source row count...")

        row_count = source_cursor.execute(
            """
            SELECT COUNT(*)
            FROM observations
            """
        ).fetchone()[0]

        print(
            f"Rows in audit database: {row_count:,}"
        )

        if row_count != EXPECTED_TOTAL_ROWS:
            raise RuntimeError(
                "Source row-count validation failed.\n"
                f"Expected: {EXPECTED_TOTAL_ROWS:,}\n"
                f"Found: {row_count:,}"
            )

        print("Source row-count validation: PASS")
        print()

        print("Checking audit database schema...")

        columns = {
            row["name"]
            for row in source_cursor.execute(
                "PRAGMA table_info(observations)"
            ).fetchall()
        }

        required_columns = {
            "id",
            "feature_hash",
            "device",
            "label",
            "source_file",
            "row_number",
        }

        missing = required_columns - columns

        if missing:
            raise RuntimeError(
                "Missing required database columns: "
                + ", ".join(sorted(missing))
            )

        print("Audit database schema validation: PASS")
        print()

        print(
            "Checking feature-vector label consistency..."
        )

        conflicts = source_cursor.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT feature_hash
                FROM observations
                GROUP BY feature_hash
                HAVING COUNT(DISTINCT label) > 1
            )
            """
        ).fetchone()[0]

        print(
            f"Cross-label feature-vector conflicts: {conflicts:,}"
        )

        if conflicts != 0:
            raise RuntimeError(
                "Feature-vector label consistency validation failed."
            )

        print("Label consistency validation: PASS")
        print()

        print("Checking original N-BaIoT labels...")

        original_labels = source_cursor.execute(
            """
            SELECT
                label,
                COUNT(*) AS count
            FROM observations
            GROUP BY label
            ORDER BY label
            """
        ).fetchall()

        for row in original_labels:

            original = str(
                row["label"]
            ).strip().lower()

            count = int(
                row["count"]
            )

            binary = normalize_label(
                original
            )

            print(
                f"  {original:10s} -> "
                f"{binary} ({label_name(binary)}) "
                f"{count:,} rows"
            )

        print("Original-label validation: PASS")
        print()

        print(
            "Building device/label duplicate groups..."
        )

        query = """
            SELECT
                device,
                feature_hash,
                label,
                COUNT(*) AS multiplicity
            FROM observations
            GROUP BY
                device,
                feature_hash,
                label
            ORDER BY
                device,
                label,
                feature_hash
        """

        device_groups = defaultdict(
            lambda: defaultdict(list)
        )

        group_metadata = {}

        total_groups = 0
        total_group_rows = 0

        for row in source_cursor.execute(query):

            device = str(
                row["device"]
            )

            feature_hash = str(
                row["feature_hash"]
            )

            binary_label = normalize_label(
                row["label"]
            )

            multiplicity = int(
                row["multiplicity"]
            )

            if multiplicity <= 0:
                raise RuntimeError(
                    f"Invalid multiplicity: {multiplicity}"
                )

            device_groups[
                device
            ][binary_label].append(
                (
                    feature_hash,
                    multiplicity,
                )
            )

            group_metadata[
                (
                    device,
                    feature_hash,
                )
            ] = (
                binary_label,
                multiplicity,
            )

            total_groups += 1
            total_group_rows += multiplicity

        print(
            f"Duplicate groups created: {total_groups:,}"
        )

        print(
            f"Rows represented by groups: {total_group_rows:,}"
        )

        if total_group_rows != EXPECTED_TOTAL_ROWS:
            raise RuntimeError(
                "Duplicate-group row total does not match "
                "source row count.\n"
                f"Expected: {EXPECTED_TOTAL_ROWS:,}\n"
                f"Found: {total_group_rows:,}"
            )

        print("Duplicate-group construction: PASS")
        print()

        print(
            "Assigning duplicate groups to "
            "train/validation/test..."
        )

        group_assignments = {}

        devices = sorted(
            device_groups.keys()
        )

        for device in devices:

            print(
                f"  Processing device: {device}"
            )

            for binary_label in (0, 1):

                groups = (
                    device_groups[
                        device
                    ].get(
                        binary_label,
                        [],
                    )
                )

                if not groups:
                    continue

                seed = make_deterministic_seed(
                    device,
                    binary_label,
                )

                assignments = (
                    assign_groups_to_partitions(
                        groups,
                        seed,
                    )
                )

                for feature_hash, split in (
                    assignments.items()
                ):

                    assignment_key = (
                        device,
                        feature_hash,
                    )

                    if assignment_key in group_assignments:

                        previous = group_assignments[
                            assignment_key
                        ]

                        if previous != split:
                            raise RuntimeError(
                                "Feature-vector group received "
                                "multiple assignments within "
                                "the same device.\n"
                                f"Device: {device}\n"
                                f"Feature hash: {feature_hash}\n"
                                f"Previous split: {previous}\n"
                                f"New split: {split}"
                            )

                    group_assignments[
                        assignment_key
                    ] = split

        print()

        print(
            "Total device/feature-vector groups assigned: "
            f"{len(group_assignments):,}"
        )

        if (
            len(group_assignments)
            != total_groups
        ):
            raise RuntimeError(
                "Not every device/feature-vector group "
                "received a split assignment."
            )

        print(
            "Group assignment count validation: PASS"
        )
        print()

        print(
            "Validating duplicate-group exclusivity..."
        )

        observation_counts = {
            "train": 0,
            "validation": 0,
            "test": 0,
        }

        observed_group_split = {}

        checked = 0

        for row in source_cursor.execute(
            """
            SELECT
                id,
                feature_hash,
                device
            FROM observations
            ORDER BY id
            """
        ):

            feature_hash = str(
                row["feature_hash"]
            )

            device = str(
                row["device"]
            )

            assignment_key = (
                device,
                feature_hash,
            )

            split = group_assignments.get(
                assignment_key
            )

            if split is None:
                raise RuntimeError(
                    "No split assignment found for "
                    "device/feature-vector group.\n"
                    f"Device: {device}\n"
                    f"Feature hash: {feature_hash}"
                )

            previous = observed_group_split.get(
                assignment_key
            )

            if (
                previous is not None
                and previous != split
            ):
                raise RuntimeError(
                    "Duplicate feature-vector group "
                    "was split across partitions within "
                    "the same device.\n"
                    f"Device: {device}\n"
                    f"Feature hash: {feature_hash}\n"
                    f"First split: {previous}\n"
                    f"New split: {split}"
                )

            observed_group_split[
                assignment_key
            ] = split

            observation_counts[
                split
            ] += 1

            checked += 1

        print(
            f"Observations checked: {checked:,}"
        )

        print(
            f"Train rows:      "
            f"{observation_counts['train']:,}"
        )

        print(
            f"Validation rows: "
            f"{observation_counts['validation']:,}"
        )

        print(
            f"Test rows:       "
            f"{observation_counts['test']:,}"
        )

        total = sum(
            observation_counts.values()
        )

        if total != EXPECTED_TOTAL_ROWS:
            raise RuntimeError(
                "Final split row count does not match source.\n"
                f"Expected: {EXPECTED_TOTAL_ROWS:,}\n"
                f"Found: {total:,}"
            )

        print(
            "Duplicate-group exclusivity validation: PASS"
        )

        print(
            "Source-to-split row-count validation: PASS"
        )

        print()

        print(
            "Creating compact group-assignment database..."
        )

        if GROUP_ASSIGNMENTS_DB.exists():
            GROUP_ASSIGNMENTS_DB.unlink()

        output_connection = sqlite3.connect(
            GROUP_ASSIGNMENTS_DB
        )

        try:

            output_cursor = (
                output_connection.cursor()
            )

            output_cursor.execute(
                """
                CREATE TABLE group_assignments (
                    device TEXT NOT NULL,
                    feature_hash TEXT NOT NULL,
                    label INTEGER NOT NULL,
                    multiplicity INTEGER NOT NULL,
                    split TEXT NOT NULL,
                    PRIMARY KEY (device, feature_hash)
                )
                """
            )

            output_cursor.execute(
                """
                CREATE INDEX idx_group_split
                ON group_assignments(split)
                """
            )

            output_cursor.execute(
                """
                CREATE INDEX idx_group_device
                ON group_assignments(device)
                """
            )

            output_cursor.execute(
                """
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

            insert_sql = """
                INSERT INTO group_assignments (
                    device,
                    feature_hash,
                    label,
                    multiplicity,
                    split
                )
                VALUES (?, ?, ?, ?, ?)
            """

            batch = []
            batch_size = 10000

            for assignment_key in group_assignments:

                device, feature_hash = (
                    assignment_key
                )

                (
                    binary_label,
                    multiplicity,
                ) = group_metadata[
                    assignment_key
                ]

                split = group_assignments[
                    assignment_key
                ]

                batch.append(
                    (
                        device,
                        feature_hash,
                        binary_label,
                        multiplicity,
                        split,
                    )
                )

                if len(batch) >= batch_size:

                    output_cursor.executemany(
                        insert_sql,
                        batch,
                    )

                    output_connection.commit()

                    batch.clear()

            if batch:

                output_cursor.executemany(
                    insert_sql,
                    batch,
                )

                output_connection.commit()

            metadata = {
                "dataset": "N-BaIoT",
                "task": "binary intrusion detection",
                "label_0": "benign",
                "label_1": "attack (Gafgyt + Mirai)",
                "number_of_clients": "9",
                "client_definition": "IoT device",
                "train_ratio": "0.70",
                "validation_ratio": "0.15",
                "test_ratio": "0.15",
                "random_seed": str(RANDOM_SEED),
                "assignment_unit": (
                    "device + feature_hash"
                ),
                "duplicate_group_policy": (
                    "identical feature-vector groups "
                    "within each device remain in one partition"
                ),
                "cross_device_duplicate_policy": (
                    "the same feature vector occurring on "
                    "different devices may receive independent "
                    "partition assignments"
                ),
                "split_scope": (
                    "within each device and binary label"
                ),
                "preprocessing_policy": (
                    "StandardScaler fitted only on training data"
                ),
                "validation_test_policy": (
                    "fixed held-out partitions across "
                    "centralized, local-only, and federated "
                    "comparisons"
                ),
                "natural_federated_condition": (
                    "device-based logical clients"
                ),
                "controlled_iid_condition": (
                    "training data redistribution only; "
                    "validation and test remain fixed"
                ),
                "formal_privacy_guarantee": "none",
            }

            output_cursor.executemany(
                """
                INSERT INTO metadata (
                    key,
                    value
                )
                VALUES (?, ?)
                """,
                metadata.items(),
            )

            output_connection.commit()

            print(
                "Compact group-assignment database: PASS"
            )

        finally:

            output_connection.close()

        print()

        print(
            "Writing duplicate-group assignment CSV..."
        )

        with open(
            DUPLICATE_ASSIGNMENTS_CSV,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.writer(
                csv_file
            )

            writer.writerow(
                [
                    "device",
                    "feature_hash",
                    "label",
                    "label_name",
                    "multiplicity",
                    "split",
                ]
            )

            for assignment_key in sorted(
                group_assignments
            ):

                device, feature_hash = (
                    assignment_key
                )

                (
                    binary_label,
                    multiplicity,
                ) = group_metadata[
                    assignment_key
                ]

                writer.writerow(
                    [
                        device,
                        feature_hash,
                        binary_label,
                        label_name(
                            binary_label
                        ),
                        multiplicity,
                        group_assignments[
                            assignment_key
                        ],
                    ]
                )

        print(
            "Duplicate-group assignment CSV: PASS"
        )

        print()

        print(
            "Writing split group summary..."
        )

        with open(
            GROUP_SUMMARY_CSV,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.writer(
                csv_file
            )

            writer.writerow(
                [
                    "device",
                    "label",
                    "label_name",
                    "split",
                    "duplicate_groups",
                    "rows",
                    "target_ratio",
                    "actual_ratio",
                    "difference_from_target",
                ]
            )

            for device in devices:

                for binary_label in (0, 1):

                    groups = (
                        device_groups[
                            device
                        ].get(
                            binary_label,
                            [],
                        )
                    )

                    if not groups:
                        continue

                    total_label_rows = sum(
                        count
                        for _, count in groups
                    )

                    for split in SPLIT_NAMES:

                        group_count = 0
                        row_count = 0

                        for (
                            feature_hash,
                            count,
                        ) in groups:

                            assignment_key = (
                                device,
                                feature_hash,
                            )

                            if (
                                group_assignments[
                                    assignment_key
                                ]
                                == split
                            ):
                                group_count += 1
                                row_count += count

                        actual_ratio = (
                            row_count
                            / total_label_rows
                            if total_label_rows
                            else 0.0
                        )

                        target_ratio = (
                            SPLIT_TARGETS[
                                split
                            ]
                        )

                        difference = (
                            actual_ratio
                            - target_ratio
                        )

                        writer.writerow(
                            [
                                device,
                                binary_label,
                                label_name(
                                    binary_label
                                ),
                                split,
                                group_count,
                                row_count,
                                f"{target_ratio:.8f}",
                                f"{actual_ratio:.8f}",
                                f"{difference:.8f}",
                            ]
                        )

        print(
            "Split group summary: PASS"
        )

        print()

        print(
            "Writing frozen split specification..."
        )

        with open(
            SPLIT_SPECIFICATION_TXT,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                "N-BaIoT DATASET SPLIT SPECIFICATION\n"
            )

            file.write(
                "==================================\n\n"
            )

            file.write(
                "Dataset: N-BaIoT\n"
            )

            file.write(
                "Task: Binary intrusion detection\n"
            )

            file.write(
                "Clients: 9 logical clients, one per IoT device\n"
            )

            file.write(
                "Label 0: Benign\n"
            )

            file.write(
                "Label 1: Attack (Gafgyt + Mirai)\n\n"
            )

            file.write(
                "Primary split:\n"
            )

            file.write(
                "  Training:   70%\n"
            )

            file.write(
                "  Validation: 15%\n"
            )

            file.write(
                "  Test:       15%\n\n"
            )

            file.write(
                "Random seed: 42\n\n"
            )

            file.write(
                "Assignment unit:\n"
            )

            file.write(
                "  (device, feature_hash)\n\n"
            )

            file.write(
                "Partitioning procedure:\n"
            )

            file.write(
                "1. Split within each IoT device.\n"
            )

            file.write(
                "2. Split benign and attack observations "
                "separately within each device.\n"
            )

            file.write(
                "3. Preserve benign/attack proportions "
                "as closely as possible.\n"
            )

            file.write(
                "4. Identical feature-vector groups within "
                "a device are treated as indivisible groups.\n"
            )

            file.write(
                "5. Duplicate groups cannot cross "
                "train/validation/test boundaries within "
                "the same device.\n"
            )

            file.write(
                "6. The same feature vector occurring on "
                "different devices may have different "
                "partition assignments.\n"
            )

            file.write(
                "7. No global deduplication is performed.\n"
            )

            file.write(
                "8. All original observations are retained.\n"
            )

            file.write(
                "9. Validation and test remain fixed "
                "across training paradigms.\n\n"
            )

            file.write(
                "Original labels:\n"
            )

            file.write(
                "  benign -> binary 0\n"
            )

            file.write(
                "  gafgyt -> binary 1\n"
            )

            file.write(
                "  mirai  -> binary 1\n\n"
            )

            file.write(
                "Training paradigms:\n"
            )

            file.write(
                "  Centralized: all device training data pooled.\n"
            )

            file.write(
                "  Local-only: each device trains independently.\n"
            )

            file.write(
                "  Natural FL: each device is a logical FL client.\n"
            )

            file.write(
                "  Controlled IID-like FL: training data may be "
                "redistributed while validation/test remain fixed.\n\n"
            )

            file.write(
                "Preprocessing:\n"
            )

            file.write(
                "  StandardScaler is fitted using training data only.\n"
            )

            file.write(
                "  Validation/test use training-fitted parameters.\n\n"
            )

            file.write(
                "Privacy statement:\n"
            )

            file.write(
                "  Federated learning avoids centralizing raw client "
                "training observations during model training. "
                "This design does not provide a formal differential "
                "privacy or secure-aggregation guarantee.\n\n"
            )

            file.write(
                "Dataset-wide duplicate audit findings:\n"
            )

            file.write(
                "  Total observations: 7,062,606\n"
            )

            file.write(
                "  Unique feature vectors: 2,482,676\n"
            )

            file.write(
                "  Duplicate rows: 4,579,930\n"
            )

            file.write(
                "  Duplicate groups: 1,891,636\n"
            )

            file.write(
                "  Cross-label feature-vector conflicts: 0\n"
            )

            file.write(
                "  Cross-device feature-vector groups: 1,871,053\n"
            )

            file.write(
                "  Cross-device + cross-label conflicts: 0\n"
            )

            file.write(
                "  Maximum feature-vector multiplicity: 36\n"
            )

        print(
            "Split specification: PASS"
        )

        print()

        print(
            "Running compact output validation..."
        )

        validation_connection = sqlite3.connect(
            GROUP_ASSIGNMENTS_DB
        )

        try:

            validation_cursor = (
                validation_connection.cursor()
            )

            database_group_count = (
                validation_cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM group_assignments
                    """
                ).fetchone()[0]
            )

            if (
                database_group_count
                != total_groups
            ):
                raise RuntimeError(
                    "Compact database group count mismatch.\n"
                    f"Expected: {total_groups:,}\n"
                    f"Found: {database_group_count:,}"
                )

            database_row_total = (
                validation_cursor.execute(
                    """
                    SELECT SUM(multiplicity)
                    FROM group_assignments
                    """
                ).fetchone()[0]
            )

            if (
                database_row_total
                != EXPECTED_TOTAL_ROWS
            ):
                raise RuntimeError(
                    "Compact database row total mismatch.\n"
                    f"Expected: {EXPECTED_TOTAL_ROWS:,}\n"
                    f"Found: {database_row_total:,}"
                )

            split_counts = {
                row[0]: row[1]
                for row in validation_cursor.execute(
                    """
                    SELECT
                        split,
                        SUM(multiplicity)
                    FROM group_assignments
                    GROUP BY split
                    """
                ).fetchall()
            }

            for split in SPLIT_NAMES:

                if (
                    split_counts.get(split, 0)
                    != observation_counts[split]
                ):
                    raise RuntimeError(
                        f"Compact database split count "
                        f"mismatch for {split}."
                    )

            duplicate_conflicts = (
                validation_cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM (
                        SELECT
                            device,
                            feature_hash
                        FROM group_assignments
                        GROUP BY
                            device,
                            feature_hash
                        HAVING COUNT(DISTINCT split) > 1
                    )
                    """
                ).fetchone()[0]
            )

            if duplicate_conflicts != 0:
                raise RuntimeError(
                    "Compact database contains "
                    "duplicate-group split conflicts."
                )

        finally:

            validation_connection.close()

        print(
            "Compact output validation: PASS"
        )

    finally:

        source_connection.close()

    print()
    print("=" * 70)
    print("SPLIT CREATION COMPLETE")
    print("=" * 70)
    print()

    print("Final row counts:")

    for split in SPLIT_NAMES:

        count = observation_counts[split]

        percentage = (
            count
            / EXPECTED_TOTAL_ROWS
        ) * 100

        print(
            f"  {split:12s}: "
            f"{count:>10,} rows "
            f"({percentage:7.3f}%)"
        )

    print()

    print("Output files:")

    print(
        f"  {GROUP_ASSIGNMENTS_DB}"
    )

    print(
        f"  {DUPLICATE_ASSIGNMENTS_CSV}"
    )

    print(
        f"  {GROUP_SUMMARY_CSV}"
    )

    print(
        f"  {SPLIT_SPECIFICATION_TXT}"
    )

    print()

    print("All validations passed.")


if __name__ == "__main__":
    main()