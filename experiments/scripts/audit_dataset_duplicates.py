import csv
import hashlib
import sqlite3
import subprocess
import tempfile
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ZIP_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "original"
    / "detection+of+iot+botnet+attacks+n+baiot.zip"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "inspection"

DB_PATH = OUTPUT_DIR / "dataset_duplicate_audit.sqlite"

SUMMARY_PATH = (
    OUTPUT_DIR / "dataset_duplicate_audit_summary.csv"
)

DEVICE_SUMMARY_PATH = (
    OUTPUT_DIR / "dataset_duplicate_audit_by_device.csv"
)

# IMPORTANT:
# All temporary extraction now happens on D:.
TEMP_ROOT = Path(r"D:\federated-iot-temp")

SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")

CHUNK_SIZE = 50_000

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


# ============================================================
# BASIC UTILITIES
# ============================================================

def log(message: str = "") -> None:
    print(message, flush=True)


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def build_source_file_path(
    csv_path: Path,
    device: str,
) -> str:
    """
    Return a stable dataset-relative source path.

    The temporary extraction directory is intentionally excluded
    so that audit metadata remains valid after the temporary
    directory is deleted.

    Example:

        Temporary path:
        D:\\federated-iot-temp\\...\\zip_extract\\
        Danmini_Doorbell\\benign_traffic.csv

        Stored source_file:
        Danmini_Doorbell\\benign_traffic.csv
    """

    device_index = None

    for index, part in enumerate(csv_path.parts):
        if part == device:
            device_index = index
            break

    if device_index is None:
        raise ValueError(
            f"Device directory not found in CSV path:\n"
            f"{csv_path}\n"
            f"Device: {device}"
        )

    relative_parts = csv_path.parts[device_index:]

    return str(Path(*relative_parts))


def verify_requirements() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"N-BaIoT ZIP archive not found:\n{ZIP_PATH}"
        )

    if not SEVEN_ZIP.exists():
        raise FileNotFoundError(
            f"7-Zip executable not found:\n{SEVEN_ZIP}"
        )

    if not TEMP_ROOT.exists():
        raise RuntimeError(
            f"Temporary directory could not be created:\n{TEMP_ROOT}"
        )


# ============================================================
# DEVICE / LABEL IDENTIFICATION
# ============================================================

def identify_device(path: Path) -> str:
    path_text = str(path).lower()

    for device in DEVICES:
        if device.lower() in path_text:
            return device

    raise ValueError(
        f"Could not identify device from path:\n{path}"
    )


def identify_label(path: Path) -> str:
    path_text = str(path).lower()

    if "benign_traffic.csv" in path_text:
        return "benign"

    if "gafgyt" in path_text:
        return "gafgyt"

    if "mirai" in path_text:
        return "mirai"

    raise ValueError(
        f"Could not identify label from path:\n{path}"
    )


# ============================================================
# FEATURE HASHING
# ============================================================

def hash_feature_vector(features: list[str]) -> str:
    """
    Hash only the 115 feature values.

    The class label is intentionally excluded.

    Therefore, if exactly the same feature vector occurs with
    different labels, the audit can detect the conflict.
    """

    payload = "\x1f".join(
        value.strip()
        for value in features
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


# ============================================================
# 7-ZIP EXTRACTION
# ============================================================

def run_7zip_extract(
    archive: Path,
    destination: Path,
) -> None:

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        str(SEVEN_ZIP),
        "x",
        str(archive),
        f"-o{destination}",
        "-y",
        "-bb1",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:

        log("")
        log("7-Zip output:")
        log(result.stdout[-5000:])

        log("")
        log("7-Zip error:")
        log(result.stderr[-5000:])

        raise RuntimeError(
            f"7-Zip failed.\n"
            f"Archive: {archive}\n"
            f"Destination: {destination}\n"
            f"Exit code: {result.returncode}"
        )


def extract_main_zip(
    zip_path: Path,
    destination: Path,
) -> None:

    log("")
    log("=" * 70)
    log("EXTRACTING ORIGINAL N-BAIoT ZIP")
    log("=" * 70)

    log(f"Source:      {zip_path}")
    log(f"Destination: {destination}")

    run_7zip_extract(
        archive=zip_path,
        destination=destination,
    )

    log("Main ZIP extraction completed.")


def extract_attack_rar(
    rar_path: Path,
    destination: Path,
) -> None:

    device = identify_device(rar_path)

    log(
        f"Extracting attack archive: "
        f"{rar_path.name} -> {device}"
    )

    run_7zip_extract(
        archive=rar_path,
        destination=destination,
    )


# ============================================================
# CSV DISCOVERY
# ============================================================

def discover_benign_csvs(
    extracted_root: Path,
) -> list[tuple[Path, str, str]]:

    results = []

    for csv_path in extracted_root.rglob("*.csv"):

        if csv_path.name.lower() != "benign_traffic.csv":
            continue

        device = identify_device(csv_path)

        results.append(
            (
                csv_path,
                device,
                "benign",
            )
        )

    return results


def discover_rar_files(
    extracted_root: Path,
) -> list[Path]:

    return sorted(
        extracted_root.rglob("*.rar")
    )


# ============================================================
# ATTACK ARCHIVE EXTRACTION
# ============================================================

def extract_all_attack_archives(
    extracted_root: Path,
    attack_root: Path,
) -> list[tuple[Path, str, str]]:

    rar_files = discover_rar_files(
        extracted_root
    )

    log("")
    log(
        f"RAR attack archives discovered: "
        f"{len(rar_files)}"
    )

    if len(rar_files) != 16:
        log(
            "WARNING: Expected 16 attack archives, "
            f"but found {len(rar_files)}."
        )

    results = []

    for index, rar_path in enumerate(
        rar_files,
        start=1,
    ):

        device = identify_device(rar_path)
        label = identify_label(rar_path)

        destination = (
            attack_root
            / device
            / rar_path.stem
        )

        log("")
        log(
            f"[{index}/{len(rar_files)}] "
            f"{device} | {label} | {rar_path.name}"
        )

        extract_attack_rar(
            rar_path=rar_path,
            destination=destination,
        )

        csv_files = sorted(
            destination.rglob("*.csv")
        )

        if not csv_files:
            raise RuntimeError(
                f"No CSV files were extracted from:\n"
                f"{rar_path}"
            )

        for csv_path in csv_files:

            if (
                csv_path.name.lower()
                == "demonstrate_structure.csv"
            ):
                continue

            results.append(
                (
                    csv_path,
                    device,
                    label,
                )
            )

    return results


# ============================================================
# SQLITE DATABASE
# ============================================================

def create_database() -> sqlite3.Connection:

    if DB_PATH.exists():
        log("")
        log(
            "Removing previous duplicate-audit database..."
        )
        DB_PATH.unlink()

    connection = sqlite3.connect(
        DB_PATH,
        timeout=300,
    )

    connection.execute(
        "PRAGMA journal_mode=WAL;"
    )

    connection.execute(
        "PRAGMA synchronous=NORMAL;"
    )

    connection.execute(
        "PRAGMA temp_store=MEMORY;"
    )

    connection.execute(
        """
        CREATE TABLE observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature_hash TEXT NOT NULL,
            device TEXT NOT NULL,
            label TEXT NOT NULL,
            source_file TEXT NOT NULL,
            row_number INTEGER NOT NULL
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_feature_hash
        ON observations(feature_hash)
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_device
        ON observations(device)
        """
    )

    connection.execute(
        """
        CREATE INDEX idx_label
        ON observations(label)
        """
    )

    connection.commit()

    return connection


# ============================================================
# CSV PROCESSING
# ============================================================

def process_csv(
    connection: sqlite3.Connection,
    csv_path: Path,
    device: str,
    label: str,
    stats: dict,
) -> None:

    log(
        f"Processing: "
        f"{device} | {label} | {csv_path.name}"
    )

    source_file = build_source_file_path(
        csv_path=csv_path,
        device=device,
    )

    with csv_path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:

        reader = csv.reader(handle)

        try:
            header = next(reader)
        except StopIteration:
            raise RuntimeError(
                f"CSV file is empty:\n{csv_path}"
            )

        if len(header) < 2:
            raise RuntimeError(
                f"Unexpected CSV structure:\n{csv_path}"
            )

        feature_count = len(header) - 1

        if stats["feature_count"] is None:

            stats["feature_count"] = (
                feature_count
            )

            stats["feature_names"] = (
                header[:-1]
            )

        elif (
            feature_count
            != stats["feature_count"]
        ):

            raise RuntimeError(
                f"Feature count mismatch:\n"
                f"{csv_path}\n"
                f"Expected: {stats['feature_count']}\n"
                f"Found: {feature_count}"
            )

        buffer = []

        row_number = 1

        for row in reader:

            if not row:
                continue

            if len(row) != len(header):

                stats["malformed_rows"] += 1

                row_number += 1
                continue

            features = row[:-1]

            feature_hash = hash_feature_vector(
                features
            )

            buffer.append(
                (
                    feature_hash,
                    device,
                    label,
                    source_file,
                    row_number,
                )
            )

            stats["rows"] += 1

            if label == "benign":
                stats["benign_rows"] += 1

            elif label == "mirai":
                stats["mirai_rows"] += 1
                stats["attack_rows"] += 1

            elif label == "gafgyt":
                stats["gafgyt_rows"] += 1
                stats["attack_rows"] += 1

            row_number += 1

            if len(buffer) >= CHUNK_SIZE:

                connection.executemany(
                    """
                    INSERT INTO observations (
                        feature_hash,
                        device,
                        label,
                        source_file,
                        row_number
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    buffer,
                )

                connection.commit()

                buffer.clear()

        if buffer:

            connection.executemany(
                """
                INSERT INTO observations (
                    feature_hash,
                    device,
                    label,
                    source_file,
                    row_number
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                buffer,
            )

            connection.commit()


# ============================================================
# DUPLICATE AUDIT
# ============================================================

def run_audit(
    connection: sqlite3.Connection,
    stats: dict,
) -> None:

    log("")
    log("=" * 70)
    log("RUNNING DUPLICATE AND LABEL-CONFLICT AUDIT")
    log("=" * 70)

    cursor = connection.cursor()

    # --------------------------------------------------------
    # Unique feature vectors
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(DISTINCT feature_hash)
        FROM observations
        """
    )

    stats["unique_feature_vectors"] = (
        cursor.fetchone()[0]
    )

    stats["duplicate_rows"] = (
        stats["rows"]
        - stats["unique_feature_vectors"]
    )

    # --------------------------------------------------------
    # Duplicate groups
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT feature_hash
            FROM observations
            GROUP BY feature_hash
            HAVING COUNT(*) > 1
        )
        """
    )

    stats["duplicate_groups"] = (
        cursor.fetchone()[0]
    )

    # --------------------------------------------------------
    # Cross-label feature vectors
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT feature_hash
            FROM observations
            GROUP BY feature_hash
            HAVING COUNT(DISTINCT label) > 1
        )
        """
    )

    stats["cross_label_feature_vectors"] = (
        cursor.fetchone()[0]
    )

    # --------------------------------------------------------
    # Cross-device feature vectors
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT feature_hash
            FROM observations
            GROUP BY feature_hash
            HAVING COUNT(DISTINCT device) > 1
        )
        """
    )

    stats["cross_device_feature_vectors"] = (
        cursor.fetchone()[0]
    )

    # --------------------------------------------------------
    # Cross-device + cross-label vectors
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT feature_hash
            FROM observations
            GROUP BY feature_hash
            HAVING COUNT(DISTINCT device) > 1
               AND COUNT(DISTINCT label) > 1
        )
        """
    )

    stats[
        "cross_device_cross_label_vectors"
    ] = cursor.fetchone()[0]

    # --------------------------------------------------------
    # Maximum multiplicity
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT MAX(cnt)
        FROM (
            SELECT
                feature_hash,
                COUNT(*) AS cnt
            FROM observations
            GROUP BY feature_hash
        )
        """
    )

    stats["maximum_duplicate_multiplicity"] = (
        cursor.fetchone()[0]
    )


# ============================================================
# DEVICE SUMMARY
# ============================================================

def write_device_summary(
    connection: sqlite3.Connection,
) -> None:

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            device,
            label,
            COUNT(*) AS rows
        FROM observations
        GROUP BY device, label
        ORDER BY device, label
        """
    )

    rows = cursor.fetchall()

    with DEVICE_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow(
            [
                "device",
                "label",
                "rows",
            ]
        )

        writer.writerows(rows)


# ============================================================
# SUMMARY
# ============================================================

def write_summary(stats: dict) -> None:

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow(
            [
                "metric",
                "value",
            ]
        )

        for key, value in stats.items():

            if key == "feature_names":
                continue

            writer.writerow(
                [
                    key,
                    value,
                ]
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    ensure_directories()
    verify_requirements()

    log("")
    log("=" * 70)
    log("N-BAIoT DATASET-WIDE DUPLICATE AND LABEL-CONFLICT AUDIT")
    log("=" * 70)

    log("")
    log(f"Project root : {PROJECT_ROOT}")
    log(f"Dataset ZIP  : {ZIP_PATH}")
    log(f"7-Zip        : {SEVEN_ZIP}")
    log(f"Temporary D: : {TEMP_ROOT}")
    log(f"Output DB    : {DB_PATH}")

    # --------------------------------------------------------
    # Temporary directory is explicitly located on D:
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="nbaiot_duplicate_",
        dir=str(TEMP_ROOT),
    ) as temp_directory:

        temp_root = Path(temp_directory)

        extracted_root = (
            temp_root / "zip_extract"
        )

        attack_root = (
            temp_root / "attack_extract"
        )

        # ----------------------------------------------------
        # Extract main ZIP
        # ----------------------------------------------------

        extract_main_zip(
            zip_path=ZIP_PATH,
            destination=extracted_root,
        )

        # ----------------------------------------------------
        # Discover benign files
        # ----------------------------------------------------

        benign_files = discover_benign_csvs(
            extracted_root
        )

        log("")
        log(
            f"Direct benign CSV files discovered: "
            f"{len(benign_files)}"
        )

        if len(benign_files) != 9:

            log(
                "WARNING: Expected 9 benign CSV files, "
                f"but found {len(benign_files)}."
            )

        # ----------------------------------------------------
        # Extract all attack RAR files
        # ----------------------------------------------------

        attack_files = (
            extract_all_attack_archives(
                extracted_root=extracted_root,
                attack_root=attack_root,
            )
        )

        log("")
        log(
            f"Attack CSV files discovered after extraction: "
            f"{len(attack_files)}"
        )

        all_files = (
            benign_files
            + attack_files
        )

        log("")
        log(
            f"Total CSV files to audit: "
            f"{len(all_files)}"
        )

        # ----------------------------------------------------
        # Create database
        # ----------------------------------------------------

        connection = create_database()

        stats = {
            "files_processed": 0,
            "rows": 0,
            "benign_rows": 0,
            "attack_rows": 0,
            "mirai_rows": 0,
            "gafgyt_rows": 0,
            "malformed_rows": 0,
            "feature_count": None,
            "unique_feature_vectors": 0,
            "duplicate_rows": 0,
            "duplicate_groups": 0,
            "cross_label_feature_vectors": 0,
            "cross_device_feature_vectors": 0,
            "cross_device_cross_label_vectors": 0,
            "maximum_duplicate_multiplicity": 0,
            "feature_names": [],
        }

        # ----------------------------------------------------
        # Process all CSV files
        # ----------------------------------------------------

        for index, (
            csv_path,
            device,
            label,
        ) in enumerate(
            all_files,
            start=1,
        ):

            log("")
            log(
                f"[{index}/{len(all_files)}]"
            )

            process_csv(
                connection=connection,
                csv_path=csv_path,
                device=device,
                label=label,
                stats=stats,
            )

            stats["files_processed"] += 1

        # ----------------------------------------------------
        # Run audit
        # ----------------------------------------------------

        run_audit(
            connection=connection,
            stats=stats,
        )

        # ----------------------------------------------------
        # Write reports
        # ----------------------------------------------------

        write_device_summary(
            connection
        )

        write_summary(
            stats
        )

        connection.close()

    # --------------------------------------------------------
    # Temporary directory is automatically deleted here
    # --------------------------------------------------------

    log("")
    log("=" * 70)
    log("AUDIT COMPLETE")
    log("=" * 70)

    log("")
    log(
        f"Files processed:              "
        f"{stats['files_processed']:,}"
    )

    log(
        f"Rows processed:               "
        f"{stats['rows']:,}"
    )

    log(
        f"Features:                     "
        f"{stats['feature_count']}"
    )

    log(
        f"Unique feature vectors:       "
        f"{stats['unique_feature_vectors']:,}"
    )

    log(
        f"Duplicate rows:               "
        f"{stats['duplicate_rows']:,}"
    )

    log(
        f"Duplicate groups:             "
        f"{stats['duplicate_groups']:,}"
    )

    log(
        f"Cross-label vectors:          "
        f"{stats['cross_label_feature_vectors']:,}"
    )

    log(
        f"Cross-device vectors:         "
        f"{stats['cross_device_feature_vectors']:,}"
    )

    log(
        f"Cross-device + cross-label:   "
        f"{stats['cross_device_cross_label_vectors']:,}"
    )

    log(
        f"Maximum duplicate multiplicity:"
        f" {stats['maximum_duplicate_multiplicity']:,}"
    )

    log("")
    log(
        f"Audit database: {DB_PATH}"
    )

    log(
        f"Summary report: {SUMMARY_PATH}"
    )

    log(
        f"Device report:  {DEVICE_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()