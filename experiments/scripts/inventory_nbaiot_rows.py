import csv
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ZIP_PATH = Path(
    r"D:\federated-iot-ids\data\raw\original\detection+of+iot+botnet+attacks+n+baiot.zip"
)

OUTPUT_DIR = Path(r"D:\federated-iot-ids\results\raw")
OUTPUT_FILE = OUTPUT_DIR / "dataset_row_inventory.csv"

DEVICES = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
]


def count_csv_rows(path: Path) -> int:
    """Count data rows without loading the CSV into memory."""
    with path.open("rb") as f:
        rows = sum(1 for _ in f)

    # One line is the header.
    return max(rows - 1, 0)


def extract_zip_member(z: zipfile.ZipFile, member: str, destination: Path) -> Path:
    """Extract one member from the main ZIP."""
    output = destination / Path(member).name

    with z.open(member) as source, output.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)

    return output


def extract_rar(rar_path: Path, destination: Path) -> list[Path]:
    """Extract all CSV files from one RAR using Windows bsdtar."""
    destination.mkdir(parents=True, exist_ok=True)

    command = [
        "tar",
        "-xf",
        str(rar_path),
        "-C",
        str(destination),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to extract {rar_path}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return sorted(destination.rglob("*.csv"))


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Dataset ZIP not found: {ZIP_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        members = set(z.namelist())

        with tempfile.TemporaryDirectory(
            prefix="nbaiot_inventory_",
            dir=str(OUTPUT_DIR),
        ) as temp_dir:

            temp_root = Path(temp_dir)

            for device_index, device in enumerate(DEVICES, start=1):
                print(f"\n[{device_index}/9] Processing: {device}")

                device_total = 0

                # -------------------------
                # Benign traffic
                # -------------------------
                benign_member = f"{device}/benign_traffic.csv"

                if benign_member in members:
                    benign_path = extract_zip_member(
                        z,
                        benign_member,
                        temp_root,
                    )

                    rows = count_csv_rows(benign_path)

                    records.append({
                        "device": device,
                        "category": "benign",
                        "attack_family": "",
                        "attack_type": "",
                        "source_file": "benign_traffic.csv",
                        "rows": rows,
                    })

                    device_total += rows

                    benign_path.unlink()

                    print(f"  benign: {rows:,} rows")
                else:
                    print("  benign: NOT FOUND")

                # -------------------------
                # Attack archives
                # -------------------------
                for family in ["mirai", "gafgyt"]:
                    rar_member = f"{device}/{family}_attacks.rar"

                    if rar_member not in members:
                        print(f"  {family}: archive not present")
                        continue

                    print(f"  {family}: extracting archive...")

                    rar_path = extract_zip_member(
                        z,
                        rar_member,
                        temp_root,
                    )

                    rar_extract_dir = temp_root / f"{device}_{family}"

                    csv_files = extract_rar(
                        rar_path,
                        rar_extract_dir,
                    )

                    family_total = 0

                    for csv_path in csv_files:
                        attack_type = csv_path.stem

                        rows = count_csv_rows(csv_path)

                        records.append({
                            "device": device,
                            "category": "attack",
                            "attack_family": family,
                            "attack_type": attack_type,
                            "source_file": csv_path.name,
                            "rows": rows,
                        })

                        family_total += rows
                        device_total += rows

                        print(
                            f"    {family}/{csv_path.name}: "
                            f"{rows:,} rows"
                        )

                    print(
                        f"  {family} total: {family_total:,} rows"
                    )

                    # Remove extracted archive and CSVs.
                    if rar_path.exists():
                        rar_path.unlink()

                    if rar_extract_dir.exists():
                        shutil.rmtree(rar_extract_dir)

                print(f"  DEVICE TOTAL: {device_total:,} rows")

    # -------------------------
    # Write inventory CSV
    # -------------------------
    fieldnames = [
        "device",
        "category",
        "attack_family",
        "attack_type",
        "source_file",
        "rows",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # -------------------------
    # Summary
    # -------------------------
    total_rows = sum(r["rows"] for r in records)
    benign_rows = sum(
        r["rows"] for r in records
        if r["category"] == "benign"
    )
    attack_rows = sum(
        r["rows"] for r in records
        if r["category"] == "attack"
    )

    print("\n" + "=" * 70)
    print("N-BaIoT ROW-COUNT INVENTORY COMPLETE")
    print("=" * 70)
    print(f"Inventory file: {OUTPUT_FILE}")
    print(f"Records counted: {len(records):,}")
    print(f"Benign rows:     {benign_rows:,}")
    print(f"Attack rows:     {attack_rows:,}")
    print(f"TOTAL rows:      {total_rows:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
