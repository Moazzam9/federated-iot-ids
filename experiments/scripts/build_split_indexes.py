from pathlib import Path
import sqlite3
import csv
import numpy as np


PROJECT_ROOT = Path(r"D:\federated-iot-ids")

AUDIT_DB = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "inspection"
    / "dataset_duplicate_audit.sqlite"
)

SPLIT_DB = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "nbaiot_group_assignments.sqlite"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "source_indexes"
)

SUMMARY_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "source_index_summary.csv"
)

SPLIT_CODES = {
    "train": 0,
    "validation": 1,
    "test": 2,
}


def get_tables(conn):
    return [
        row[0]
        for row in conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()
    ]


def get_columns(conn, table_name):
    return [
        row[1]
        for row in conn.execute(
            f'PRAGMA table_info("{table_name}")'
        ).fetchall()
    ]


def find_assignment_table(conn):
    """
    Find the compact split-assignment table.

    The table must contain:
        device
        feature_hash
        split
    """

    candidates = []

    for table in get_tables(conn):
        columns = set(get_columns(conn, table))

        required = {
            "device",
            "feature_hash",
            "split",
        }

        if required.issubset(columns):
            candidates.append(table)

    if len(candidates) != 1:
        raise RuntimeError(
            "Could not uniquely identify the split-assignment table.\n"
            f"Candidates found: {candidates}"
        )

    return candidates[0]


def validate_paths():
    print("=" * 70)
    print("PATH VALIDATION")
    print("=" * 70)

    required_files = [
        (
            AUDIT_DB,
            "Original dataset audit database",
        ),
        (
            SPLIT_DB,
            "Frozen split assignment database",
        ),
    ]

    for path, description in required_files:
        print(f"{description}:")
        print(f"  {path}")
        print(f"  Exists: {path.exists()}")

        if not path.exists():
            raise FileNotFoundError(
                f"Required file does not exist:\n{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Output directory:")
    print(f"  {OUTPUT_DIR}")
    print(f"  Exists: {OUTPUT_DIR.exists()}")


def inspect_assignment_database():
    print()
    print("=" * 70)
    print("SPLIT ASSIGNMENT DATABASE")
    print("=" * 70)

    with sqlite3.connect(SPLIT_DB) as conn:

        tables = get_tables(conn)

        print("Tables:")
        for table in tables:
            print(f"  - {table}")

        assignment_table = find_assignment_table(conn)

        columns = get_columns(
            conn,
            assignment_table,
        )

        print()
        print(f"Assignment table: {assignment_table}")
        print(f"Columns: {columns}")

        counts = conn.execute(
            f"""
            SELECT
                split,
                COUNT(*)
            FROM "{assignment_table}"
            GROUP BY split
            ORDER BY split
            """
        ).fetchall()

        print()
        print("Assignment groups by split:")

        for split, count in counts:
            print(
                f"  {split:12} {count:,}"
            )

        return assignment_table


def attach_split_database(audit_conn):
    """
    Attach the frozen split database to the audit database
    connection.

    This allows us to perform:

        observations
            JOIN
        splitdb.group_assignments

    without copying the 6.9 million assignment rows.
    """

    audit_conn.execute(
        "ATTACH DATABASE ? AS splitdb",
        (str(SPLIT_DB),),
    )


def build_indexes(assignment_table):

    print()
    print("=" * 70)
    print("BUILDING SOURCE SPLIT INDEXES")
    print("=" * 70)

    source_rows = []

    with sqlite3.connect(AUDIT_DB) as audit_conn:

        audit_conn.execute(
            "PRAGMA cache_size = -200000"
        )

        # Make the frozen split database available to
        # the audit database connection.
        attach_split_database(
            audit_conn
        )

        total_rows = audit_conn.execute(
            """
            SELECT COUNT(*)
            FROM observations
            """
        ).fetchone()[0]

        print(
            f"Audit database rows: {total_rows:,}"
        )

        source_files = audit_conn.execute(
            """
            SELECT
                source_file,
                COUNT(*) AS row_count,
                MIN(row_number) AS first_row,
                MAX(row_number) AS last_row
            FROM observations
            GROUP BY source_file
            ORDER BY source_file
            """
        ).fetchall()

        print(
            f"Source files: {len(source_files)}"
        )

        # --------------------------------------------------
        # Verify source row numbering.
        # --------------------------------------------------

        for (
            source_file,
            row_count,
            first_row,
            last_row,
        ) in source_files:

            if first_row != 1:
                raise RuntimeError(
                    "Invalid first row number detected:\n"
                    f"  source_file={source_file}\n"
                    f"  first_row={first_row}\n"
                )

            if last_row != row_count:
                raise RuntimeError(
                    "Invalid final row number detected:\n"
                    f"  source_file={source_file}\n"
                    f"  row_count={row_count}\n"
                    f"  last_row={last_row}\n"
                )

        print(
            "Source row-number validation: PASS"
        )

        # --------------------------------------------------
        # Process each source file independently.
        # --------------------------------------------------

        for file_number, (
            source_file,
            row_count,
            first_row,
            last_row,
        ) in enumerate(
            source_files,
            start=1,
        ):

            print()
            print(
                f"[{file_number}/{len(source_files)}] "
                f"Processing: {source_file}"
            )

            print(
                f"  Rows: {row_count:,}"
            )

            # One byte per observation:
            #
            #   0 = train
            #   1 = validation
            #   2 = test
            #
            # 255 means "unassigned" and is used
            # temporarily for validation.
            index_array = np.full(
                row_count,
                255,
                dtype=np.uint8,
            )

            # --------------------------------------------------
            # IMPORTANT:
            #
            # observations is in the audit database.
            #
            # assignment table is in the attached split
            # database, referenced as splitdb.<table>.
            #
            # The JOIN key is:
            #
            #     (device, feature_hash)
            #
            # NOT feature_hash alone.
            # --------------------------------------------------

            query = f"""
                SELECT
                    o.row_number,
                    g.split
                FROM observations AS o
                INNER JOIN splitdb."{assignment_table}" AS g
                    ON o.device = g.device
                    AND o.feature_hash = g.feature_hash
                WHERE o.source_file = ?
                ORDER BY o.row_number
            """

            matched = 0

            for (
                row_number,
                split,
            ) in audit_conn.execute(
                query,
                (source_file,),
            ):

                if split not in SPLIT_CODES:
                    raise RuntimeError(
                        "Unexpected split value:\n"
                        f"  source_file={source_file}\n"
                        f"  row_number={row_number}\n"
                        f"  split={split!r}"
                    )

                if not (
                    1 <= row_number <= row_count
                ):
                    raise RuntimeError(
                        "Invalid row number:\n"
                        f"  source_file={source_file}\n"
                        f"  row_number={row_number}\n"
                        f"  row_count={row_count}"
                    )

                index_array[
                    row_number - 1
                ] = SPLIT_CODES[split]

                matched += 1

            # --------------------------------------------------
            # Verify every source observation received
            # exactly one split assignment.
            # --------------------------------------------------

            if matched != row_count:
                raise RuntimeError(
                    "Split mapping incomplete:\n"
                    f"  source_file={source_file}\n"
                    f"  expected rows={row_count:,}\n"
                    f"  matched rows={matched:,}"
                )

            unassigned = int(
                np.sum(
                    index_array == 255
                )
            )

            if unassigned != 0:
                raise RuntimeError(
                    "Unassigned observations detected:\n"
                    f"  source_file={source_file}\n"
                    f"  unassigned={unassigned:,}"
                )

            train_count = int(
                np.sum(index_array == 0)
            )

            validation_count = int(
                np.sum(index_array == 1)
            )

            test_count = int(
                np.sum(index_array == 2)
            )

            if (
                train_count
                + validation_count
                + test_count
                != row_count
            ):
                raise RuntimeError(
                    "Split counts do not equal "
                    "source row count:\n"
                    f"  source_file={source_file}\n"
                    f"  train={train_count:,}\n"
                    f"  validation={validation_count:,}\n"
                    f"  test={test_count:,}\n"
                    f"  total={row_count:,}"
                )

            # --------------------------------------------------
            # Save compact binary index.
            # --------------------------------------------------

            output_name = (
                f"source_{file_number:03d}.npy"
            )

            output_path = (
                OUTPUT_DIR
                / output_name
            )

            np.save(
                output_path,
                index_array,
            )

            # --------------------------------------------------
            # Verify saved file.
            # --------------------------------------------------

            reloaded = np.load(
                output_path,
                mmap_mode="r",
            )

            if len(reloaded) != row_count:
                raise RuntimeError(
                    "Saved index length mismatch:\n"
                    f"  source_file={source_file}\n"
                    f"  expected={row_count:,}\n"
                    f"  actual={len(reloaded):,}"
                )

            if not np.array_equal(
                np.asarray(reloaded),
                index_array,
            ):
                raise RuntimeError(
                    "Saved index content verification failed:\n"
                    f"  source_file={source_file}"
                )

            source_rows.append(
                {
                    "source_index": file_number,
                    "source_file": source_file,
                    "index_file": output_name,
                    "row_count": row_count,
                    "train_rows": train_count,
                    "validation_rows": validation_count,
                    "test_rows": test_count,
                }
            )

            print(
                f"  Train:      {train_count:,}"
            )

            print(
                f"  Validation: {validation_count:,}"
            )

            print(
                f"  Test:       {test_count:,}"
            )

            print(
                f"  Index:      {output_name}"
            )

            print(
                "  Verification: PASS"
            )

    return source_rows


def write_summary(source_rows):

    print()
    print("=" * 70)
    print("WRITING SOURCE INDEX SUMMARY")
    print("=" * 70)

    fieldnames = [
        "source_index",
        "source_file",
        "index_file",
        "row_count",
        "train_rows",
        "validation_rows",
        "test_rows",
    ]

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(source_rows)

    print(
        "Summary written to:"
    )

    print(
        f"  {SUMMARY_CSV}"
    )


def validate_global_counts(
    source_rows
):

    print()
    print("=" * 70)
    print("GLOBAL SPLIT VALIDATION")
    print("=" * 70)

    total = sum(
        row["row_count"]
        for row in source_rows
    )

    train = sum(
        row["train_rows"]
        for row in source_rows
    )

    validation = sum(
        row["validation_rows"]
        for row in source_rows
    )

    test = sum(
        row["test_rows"]
        for row in source_rows
    )

    print(
        f"Total rows:      {total:,}"
    )

    print(
        f"Train rows:      {train:,}"
    )

    print(
        f"Validation rows: {validation:,}"
    )

    print(
        f"Test rows:       {test:,}"
    )

    print()

    print(
        f"Train %:         "
        f"{train / total * 100:.6f}%"
    )

    print(
        f"Validation %:    "
        f"{validation / total * 100:.6f}%"
    )

    print(
        f"Test %:          "
        f"{test / total * 100:.6f}%"
    )

    # These are the authoritative counts from
    # Stage 5G.20C.
    expected_total = 7_062_606
    expected_train = 4_943_824
    expected_validation = 1_059_394
    expected_test = 1_059_388

    if total != expected_total:
        raise RuntimeError(
            "Global total mismatch:\n"
            f"  expected={expected_total:,}\n"
            f"  actual={total:,}"
        )

    if train != expected_train:
        raise RuntimeError(
            "Global train count mismatch:\n"
            f"  expected={expected_train:,}\n"
            f"  actual={train:,}"
        )

    if validation != expected_validation:
        raise RuntimeError(
            "Global validation count mismatch:\n"
            f"  expected={expected_validation:,}\n"
            f"  actual={validation:,}"
        )

    if test != expected_test:
        raise RuntimeError(
            "Global test count mismatch:\n"
            f"  expected={expected_test:,}\n"
            f"  actual={test:,}"
        )

    if (
        total
        != train
        + validation
        + test
    ):
        raise RuntimeError(
            "Global split counts do not sum "
            "to total rows."
        )

    print()
    print(
        "Global split-count validation: PASS"
    )


def main():

    print()
    print("#" * 70)
    print(
        "N-BAIoT COMPACT SPLIT INDEX BUILDER"
    )
    print("#" * 70)

    validate_paths()

    assignment_table = (
        inspect_assignment_database()
    )

    source_rows = build_indexes(
        assignment_table
    )

    write_summary(
        source_rows
    )

    validate_global_counts(
        source_rows
    )

    print()
    print("#" * 70)
    print(
        "STAGE 5G.20D.2 COMPLETE"
    )
    print("#" * 70)

    print()
    print("Created:")

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        f"  {SUMMARY_CSV}"
    )

    print()
    print("Split codes:")

    print(
        "  0 = train"
    )

    print(
        "  1 = validation"
    )

    print(
        "  2 = test"
    )

    print()
    print(
        "All source files were mapped "
        "and verified."
    )


if __name__ == "__main__":
    main()