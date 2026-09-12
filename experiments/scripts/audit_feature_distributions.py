from __future__ import annotations

import csv
import io
import math
import tarfile
import zipfile
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(r"D:\federated-iot-ids")

ARCHIVE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "original"
    / "detection+of+iot+botnet+attacks+n+baiot.zip"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "inspection"

OUTPUT_FILE = OUTPUT_DIR / "feature_distribution_audit.csv"

CHUNK_SIZE = 10_000
RESERVOIR_SIZE = 50_000
RANDOM_SEED = 20260907

EXPECTED_FEATURES = 115


def is_csv(name: str) -> bool:
    return name.lower().endswith(".csv")


def process_rows(
    rows,
    header,
    feature_names,
    count,
    minimum,
    maximum,
    mean,
    m2,
    reservoir,
    reservoir_seen,
    rng,
):
    if feature_names is None:
        feature_names = header

    if len(header) != EXPECTED_FEATURES:
        return feature_names

    if header != feature_names:
        raise ValueError(
            "Feature header mismatch detected.\n"
            f"Expected first header with {len(feature_names)} columns.\n"
            f"Found header with {len(header)} columns."
        )

    values = np.asarray(rows, dtype=np.float64)

    rows_count, columns = values.shape

    for j in range(columns):

        column = values[:, j]

        finite = np.isfinite(column)

        if not np.any(finite):
            continue

        x = column[finite]

        n_old = count[j]
        n_batch = len(x)
        n_new = n_old + n_batch

        batch_mean = float(np.mean(x))
        batch_m2 = float(
            np.sum((x - batch_mean) ** 2)
        )

        if n_old == 0:
            mean[j] = batch_mean
            m2[j] = batch_m2
        else:
            delta = batch_mean - mean[j]

            mean[j] = (
                mean[j]
                + delta * n_batch / n_new
            )

            m2[j] = (
                m2[j]
                + batch_m2
                + delta * delta
                * n_old
                * n_batch
                / n_new
            )

        count[j] = n_new

        minimum[j] = min(
            minimum[j],
            float(np.min(x)),
        )

        maximum[j] = max(
            maximum[j],
            float(np.max(x)),
        )

        # Reservoir sampling for approximate percentiles.
        for value in x:

            reservoir_seen[j] += 1

            seen = reservoir_seen[j]

            if len(reservoir[j]) < RESERVOIR_SIZE:

                reservoir[j].append(
                    float(value)
                )

            else:

                replacement = rng.integers(
                    0,
                    seen,
                )

                if replacement < RESERVOIR_SIZE:

                    reservoir[j][replacement] = (
                        float(value)
                    )

    return feature_names


def read_csv_stream(
    file_object,
):
    text = io.TextIOWrapper(
        file_object,
        encoding="utf-8",
        errors="replace",
    )

    reader = csv.reader(text)

    try:
        header = next(reader)
    except StopIteration:
        return

    if len(header) != EXPECTED_FEATURES:
        return

    rows = []

    for row in reader:

        if not row:
            continue

        if len(row) != EXPECTED_FEATURES:
            continue

        rows.append(row)

        if len(rows) >= CHUNK_SIZE:

            yield header, rows

            rows = []

    if rows:
        yield header, rows


def process_csv(
    file_object,
    source_name,
    feature_names,
    count,
    minimum,
    maximum,
    mean,
    m2,
    reservoir,
    reservoir_seen,
    rng,
):
    rows_processed = 0
    chunks_processed = 0

    for header, rows in read_csv_stream(
        file_object
    ):

        feature_names = process_rows(
            rows,
            header,
            feature_names,
            count,
            minimum,
            maximum,
            mean,
            m2,
            reservoir,
            reservoir_seen,
            rng,
        )

        rows_processed += len(rows)
        chunks_processed += 1

    return (
        feature_names,
        rows_processed,
        chunks_processed,
    )


def main():

    if not ARCHIVE_PATH.exists():

        raise FileNotFoundError(
            f"Dataset archive not found:\n{ARCHIVE_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    count = np.zeros(
        EXPECTED_FEATURES,
        dtype=np.int64,
    )

    minimum = np.full(
        EXPECTED_FEATURES,
        np.inf,
        dtype=np.float64,
    )

    maximum = np.full(
        EXPECTED_FEATURES,
        -np.inf,
        dtype=np.float64,
    )

    mean = np.zeros(
        EXPECTED_FEATURES,
        dtype=np.float64,
    )

    m2 = np.zeros(
        EXPECTED_FEATURES,
        dtype=np.float64,
    )

    reservoir = [
        []
        for _ in range(EXPECTED_FEATURES)
    ]

    reservoir_seen = np.zeros(
        EXPECTED_FEATURES,
        dtype=np.int64,
    )

    feature_names = None

    files_processed = 0
    chunks_processed = 0
    rows_processed = 0

    print("=" * 80)
    print("N-BaIoT COMPLETE FEATURE DISTRIBUTION AUDIT")
    print("=" * 80)

    print(
        f"Archive: {ARCHIVE_PATH}"
    )

    print(
        f"Output:  {OUTPUT_FILE}"
    )

    print()

    with zipfile.ZipFile(
        ARCHIVE_PATH,
        "r",
    ) as archive:

        members = archive.infolist()

        direct_csvs = [
            info
            for info in members
            if not info.is_dir()
            and is_csv(info.filename)
            and "demonstrate_structure.csv"
            not in info.filename.lower()
        ]

        rar_files = [
            info
            for info in members
            if not info.is_dir()
            and info.filename.lower().endswith(".rar")
        ]

        print(
            f"Direct benign CSV files: {len(direct_csvs)}"
        )

        print(
            f"RAR archives containing attacks: {len(rar_files)}"
        )

        print()

        # ---------------------------------------------------------
        # DIRECT CSV FILES
        # ---------------------------------------------------------

        for index, info in enumerate(
            direct_csvs,
            start=1,
        ):

            print(
                f"[BENIGN {index:02d}/{len(direct_csvs):02d}] "
                f"{info.filename}"
            )

            with archive.open(info) as file_object:

                (
                    feature_names,
                    file_rows,
                    file_chunks,
                ) = process_csv(
                    file_object,
                    info.filename,
                    feature_names,
                    count,
                    minimum,
                    maximum,
                    mean,
                    m2,
                    reservoir,
                    reservoir_seen,
                    rng,
                )

            files_processed += 1
            chunks_processed += file_chunks
            rows_processed += file_rows

            print(
                f"    rows={file_rows:,}"
            )

        # ---------------------------------------------------------
        # RAR ARCHIVES
        # ---------------------------------------------------------

        for index, info in enumerate(
            rar_files,
            start=1,
        ):

            print()
            print(
                f"[ATTACK RAR {index:02d}/{len(rar_files):02d}] "
                f"{info.filename}"
            )

            # Extract only this RAR temporarily.
            temporary_rar = (
                OUTPUT_DIR
                / "_temporary_attack_archive.rar"
            )

            with archive.open(info) as source, \
                 temporary_rar.open("wb") as target:

                while True:

                    chunk = source.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    target.write(chunk)

            # Python's standard library cannot directly read RAR.
            # Windows tar/bsdtar is available in the environment,
            # so we use the tar command through PowerShell-compatible
            # extraction performed by the operating system.

            import subprocess
            import shutil

            temporary_dir = (
                OUTPUT_DIR
                / "_temporary_attack_extracted"
            )

            if temporary_dir.exists():
                shutil.rmtree(
                    temporary_dir
                )

            temporary_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            command = [
                "tar",
                "-xf",
                str(temporary_rar),
                "-C",
                str(temporary_dir),
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:

                raise RuntimeError(
                    "RAR extraction failed.\n"
                    f"Archive: {info.filename}\n"
                    f"STDOUT:\n{result.stdout}\n"
                    f"STDERR:\n{result.stderr}"
                )

            extracted_csvs = list(
                temporary_dir.rglob("*.csv")
            )

            print(
                f"    attack CSV files found: "
                f"{len(extracted_csvs)}"
            )

            for csv_path in extracted_csvs:

                print(
                    f"    processing: "
                    f"{csv_path.name}"
                )

                with csv_path.open(
                    "rb"
                ) as file_object:

                    (
                        feature_names,
                        file_rows,
                        file_chunks,
                    ) = process_csv(
                        file_object,
                        str(csv_path),
                        feature_names,
                        count,
                        minimum,
                        maximum,
                        mean,
                        m2,
                        reservoir,
                        reservoir_seen,
                        rng,
                    )

                files_processed += 1
                chunks_processed += file_chunks
                rows_processed += file_rows

                print(
                    f"        rows={file_rows:,}"
                )

            temporary_rar.unlink(
                missing_ok=True
            )

            shutil.rmtree(
                temporary_dir,
                ignore_errors=True,
            )

    # -------------------------------------------------------------
    # FINAL STATISTICS
    # -------------------------------------------------------------

    if feature_names is None:

        raise RuntimeError(
            "No valid feature CSV files were processed."
        )

    print()
    print("=" * 80)
    print("CALCULATING FINAL STATISTICS")
    print("=" * 80)

    results = []

    for j, feature in enumerate(
        feature_names
    ):

        n = int(count[j])

        if n == 0:

            variance = float("nan")
            std = float("nan")

            percentiles = [
                float("nan")
            ] * 7

        else:

            variance = (
                m2[j] / n
            )

            std = math.sqrt(
                max(
                    variance,
                    0.0,
                )
            )

            sample = np.asarray(
                reservoir[j],
                dtype=np.float64,
            )

            percentiles = np.percentile(
                sample,
                [
                    1,
                    5,
                    25,
                    50,
                    75,
                    95,
                    99,
                ],
            )

        (
            q01,
            q05,
            q25,
            q50,
            q75,
            q95,
            q99,
        ) = percentiles

        feature_range = (
            maximum[j] - minimum[j]
            if n > 0
            else float("nan")
        )

        results.append(
            {
                "feature_index": j + 1,
                "feature_name": feature,
                "count": n,
                "minimum": minimum[j],
                "q01_approx": q01,
                "q05_approx": q05,
                "q25_approx": q25,
                "median_approx": q50,
                "q75_approx": q75,
                "q95_approx": q95,
                "q99_approx": q99,
                "maximum": maximum[j],
                "range": feature_range,
                "mean": mean[j],
                "std": std,
                "zero_variance": (
                    std == 0.0
                ),
            }
        )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print(
        f"Files processed:  {files_processed}"
    )

    print(
        f"Chunks processed: {chunks_processed}"
    )

    print(
        f"Rows processed:   {rows_processed:,}"
    )

    print(
        f"Features:         {len(results)}"
    )

    print()

    print(
        f"Audit written to:\n{OUTPUT_FILE}"
    )

    print()
    print("=" * 80)
    print("FEATURES WITH LARGEST RANGES")
    print("=" * 80)

    largest_range = sorted(
        results,
        key=lambda x: x["range"],
        reverse=True,
    )[:15]

    for row in largest_range:

        print(
            f'{row["feature_index"]:3d} '
            f'{row["feature_name"]:<35} '
            f'range={row["range"]:.6e} '
            f'min={row["minimum"]:.6e} '
            f'max={row["maximum"]:.6e}'
        )

    print()
    print("=" * 80)
    print("ZERO-VARIANCE FEATURES")
    print("=" * 80)

    zero_variance = [
        row
        for row in results
        if row["zero_variance"]
    ]

    if zero_variance:

        for row in zero_variance:

            print(
                f'{row["feature_index"]:3d} '
                f'{row["feature_name"]}'
            )

    else:

        print(
            "None detected."
        )

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()