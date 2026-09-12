from pathlib import Path
import sqlite3


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


def inspect_database(db_path: Path, name: str) -> None:
    print("=" * 70)
    print(f"{name}")
    print("=" * 70)
    print(f"Path: {db_path}")
    print(f"Exists: {db_path.exists()}")

    if not db_path.exists():
        print()
        return

    with sqlite3.connect(db_path) as conn:
        tables = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()

        print("\nTables:")
        for (table_name,) in tables:
            print(f"  - {table_name}")

            columns = conn.execute(
                f"PRAGMA table_info({table_name})"
            ).fetchall()

            print("    Columns:")
            for column in columns:
                print(
                    f"      {column[1]} "
                    f"(type={column[2]}, not_null={column[3]})"
                )


def main() -> None:
    inspect_database(AUDIT_DB, "ORIGINAL DATASET AUDIT DATABASE")
    print()
    inspect_database(SPLIT_DB, "COMPACT SPLIT ASSIGNMENT DATABASE")

    print()
    print("=" * 70)
    print("SAMPLE SOURCE FILE RECORDS")
    print("=" * 70)

    with sqlite3.connect(AUDIT_DB) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }

        if "observations" not in tables:
            print("ERROR: observations table was not found.")
            return

        rows = conn.execute(
            """
            SELECT *
            FROM observations
            LIMIT 10
            """
        ).fetchall()

        columns = [
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(observations)"
            ).fetchall()
        ]

        print("\nColumns:")
        print(columns)

        print("\nFirst 10 observations:")
        for row in rows:
            print(row)

        print()
        print("=" * 70)
        print("SOURCE FILE SUMMARY")
        print("=" * 70)

        summary = conn.execute(
            """
            SELECT
                source_file,
                COUNT(*) AS row_count,
                MIN(row_number) AS first_row_number,
                MAX(row_number) AS last_row_number
            FROM observations
            GROUP BY source_file
            ORDER BY source_file
            """
        ).fetchall()

        print(
            f"{'Source file':55} "
            f"{'Rows':>10} "
            f"{'First':>10} "
            f"{'Last':>10}"
        )
        print("-" * 90)

        for source_file, row_count, first_row, last_row in summary:
            print(
                f"{str(source_file):55} "
                f"{row_count:10,} "
                f"{first_row:10,} "
                f"{last_row:10,}"
            )

        print()
        print(f"Source files found: {len(summary)}")
        print(f"Total rows represented: {sum(row[1] for row in summary):,}")


if __name__ == "__main__":
    main()