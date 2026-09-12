from pathlib import Path
import pandas as pd


FILES = {
    "danmini_benign": Path(
        r"D:\federated-iot-ids\data\raw\inspection\Danmini_Doorbell_benign_traffic.csv"
    ),
    "danmini_mirai_ack": Path(
        r"D:\federated-iot-ids\data\raw\inspection\mirai_ack\ack.csv"
    ),
    "danmini_gafgyt_combo": Path(
        r"D:\federated-iot-ids\data\raw\inspection\gafgyt_combo\combo.csv"
    ),
}


def audit_file(name: str, path: Path) -> None:
    print("=" * 80)
    print(f"FILE: {name}")
    print(f"PATH: {path}")
    print("=" * 80)

    if not path.exists():
        print("STATUS: FILE NOT FOUND")
        print()
        return

    df = pd.read_csv(path)

    total_rows = len(df)
    unique_rows = len(df.drop_duplicates())
    duplicate_rows = total_rows - unique_rows

    duplicate_percentage = (
        (duplicate_rows / total_rows) * 100 if total_rows else 0.0
    )

    duplicated_mask = df.duplicated(keep=False)
    rows_in_duplicate_groups = int(duplicated_mask.sum())

    number_of_duplicate_groups = (
        df.loc[duplicated_mask]
        .groupby(list(df.columns), sort=False)
        .size()
        .shape[0]
        if rows_in_duplicate_groups
        else 0
    )

    print(f"Rows:                    {total_rows:,}")
    print(f"Unique rows:             {unique_rows:,}")
    print(f"Duplicate rows:          {duplicate_rows:,}")
    print(f"Duplicate percentage:    {duplicate_percentage:.4f}%")
    print(f"Rows in duplicate groups:{rows_in_duplicate_groups:,}")
    print(f"Duplicate groups:        {number_of_duplicate_groups:,}")
    print()


def main() -> None:
    print("N-BaIoT DUPLICATE STRUCTURE AUDIT")
    print()

    for name, path in FILES.items():
        audit_file(name, path)

    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()