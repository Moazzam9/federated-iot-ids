from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

SPLIT_CODES = {
    "train": 0,
    "validation": 1,
    "test": 2,
}

BINARY_LABEL_MAP = {
    "benign": 0,
    "gafgyt": 1,
    "mirai": 1,
}

LABEL_COLUMN = "binary_label"

NON_FEATURE_COLUMNS = {
    "label",
    "binary_label",
    "attack_family",
    "source_file",
    "row_number",
    "feature_hash",
}


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SourceRecord:
    """
    Metadata describing one original N-BaIoT source file.
    """

    source_index: int
    source_file: Path
    index_file: Path
    row_count: int
    train_rows: int
    validation_rows: int
    test_rows: int


@dataclass(frozen=True)
class DataChunk:
    """
    One memory-bounded chunk returned by the loader.
    """

    features: pd.DataFrame
    binary_labels: np.ndarray
    attack_families: np.ndarray
    source_file: Path
    row_numbers: np.ndarray


# ----------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------

class NBaIoTSplitLoader:
    """
    Memory-safe access to the leakage-safe N-BaIoT split.

    The loader does NOT create copies of the complete dataset.

    It reads original CSV files in chunks and uses the compact
    per-source .npy split index to select train, validation, or test
    observations.

    Split codes:
        0 = train
        1 = validation
        2 = test
    """

    def __init__(
        self,
        project_root: str | Path,
        data_root: str | Path | None = None,
        summary_file: str | Path | None = None,
        chunk_size: int = 10_000,
    ) -> None:

        self.project_root = Path(project_root)

        if data_root is None:
            data_root = self.project_root.parent / "federated-iot-temp"

        self.data_root = Path(data_root)

        if summary_file is None:
            summary_file = (
                self.project_root
                / "data"
                / "processed"
                / "splits"
                / "source_index_summary.csv"
            )

        self.summary_file = Path(summary_file)

        self.chunk_size = int(chunk_size)

        if self.chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        self.records = self._load_source_summary()

        if not self.records:
            raise RuntimeError(
                "No source records were found."
            )

    # ------------------------------------------------------------------
    # Summary loading
    # ------------------------------------------------------------------

    def _load_source_summary(self) -> list[SourceRecord]:

        if not self.summary_file.exists():
            raise FileNotFoundError(
                "Source index summary was not found:\n"
                f"{self.summary_file}"
            )

        summary = pd.read_csv(
            self.summary_file
        )

        required_columns = {
            "source_index",
            "source_file",
            "index_file",
            "row_count",
            "train_rows",
            "validation_rows",
            "test_rows",
        }

        missing = (
            required_columns
            - set(summary.columns)
        )

        if missing:
            raise ValueError(
                "Source index summary is missing columns:\n"
                f"{sorted(missing)}"
            )

        records = []

        for row in summary.itertuples(
            index=False
        ):

            source_index = int(
                row.source_index
            )

            source_file = Path(
                str(row.source_file)
            )

            index_file = Path(
                str(row.index_file)
            )

            # The summary stores the absolute source path.
            # Therefore use it directly when absolute.
            #
            # Relative paths are resolved relative to data_root.
            if not source_file.is_absolute():
                source_file = (
                    self.data_root
                    / source_file
                )

            if not index_file.is_absolute():
                index_file = (
                    self.project_root
                    / "data"
                    / "processed"
                    / "splits"
                    / "source_indexes"
                    / index_file
                )

            records.append(
                SourceRecord(
                    source_index=source_index,
                    source_file=source_file,
                    index_file=index_file,
                    row_count=int(
                        row.row_count
                    ),
                    train_rows=int(
                        row.train_rows
                    ),
                    validation_rows=int(
                        row.validation_rows
                    ),
                    test_rows=int(
                        row.test_rows
                    ),
                )
            )

        records.sort(
            key=lambda record:
            record.source_index
        )

        return records

    # ------------------------------------------------------------------
    # Public information methods
    # ------------------------------------------------------------------

    def source_count(self) -> int:
        return len(self.records)

    def total_rows(self) -> int:
        return sum(
            record.row_count
            for record in self.records
        )

    def total_rows_for_split(
        self,
        split: str,
    ) -> int:

        self._validate_split(
            split
        )

        return sum(
            getattr(
                record,
                f"{split}_rows"
            )
            for record in self.records
        )

    def list_devices(self) -> list[str]:

        devices = set()

        for record in self.records:
            devices.add(
                self._extract_device(
                    record.source_file
                )
            )

        return sorted(devices)

    def records_for_device(
        self,
        device: str,
    ) -> list[SourceRecord]:

        matches = []

        for record in self.records:

            record_device = (
                self._extract_device(
                    record.source_file
                )
            )

            if record_device == device:
                matches.append(record)

        return matches

    # ------------------------------------------------------------------
    # Source-file metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_device(
        source_file: Path,
    ) -> str:

        # N-BaIoT source files are located under a
        # device directory.
        #
        # Example:
        #
        # ...\Danmini_Doorbell\benign_traffic.csv
        #
        # or:
        #
        # ...\Danmini_Doorbell\gafgyt_attacks\combo.csv

        parts = source_file.parts

        if (
            source_file.name
            == "benign_traffic.csv"
        ):

            if len(parts) < 2:
                raise ValueError(
                    f"Cannot determine device from: "
                    f"{source_file}"
                )

            return parts[-2]

        # Attack files have one additional
        # directory such as gafgyt_attacks.
        if len(parts) < 3:
            raise ValueError(
                f"Cannot determine device from: "
                f"{source_file}"
            )

        return parts[-3]

    @staticmethod
    def _extract_attack_family(
        source_file: Path,
    ) -> str:

        parts_lower = [
            part.lower()
            for part in source_file.parts
        ]

        if "gafgyt_attacks" in parts_lower:
            return "gafgyt"

        if "mirai_attacks" in parts_lower:
            return "mirai"

        if source_file.name.lower() == "benign_traffic.csv":
            return "benign"

        raise ValueError(
            "Could not determine attack family from source file:\n"
            f"{source_file}"
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_split(
        split: str,
    ) -> None:

        if split not in SPLIT_CODES:
            raise ValueError(
                "Invalid split.\n"
                f"Expected one of: "
                f"{list(SPLIT_CODES)}\n"
                f"Received: {split!r}"
            )

    def validate_source_indexes(
        self,
    ) -> None:

        print(
            "Validating source split indexes..."
        )

        for record in self.records:

            if not record.source_file.exists():
                raise FileNotFoundError(
                    "Source CSV does not exist:\n"
                    f"{record.source_file}"
                )

            if not record.index_file.exists():
                raise FileNotFoundError(
                    "Split index does not exist:\n"
                    f"{record.index_file}"
                )

            index = np.load(
                record.index_file,
                mmap_mode="r",
            )

            if len(index) != record.row_count:
                raise ValueError(
                    "Split index length mismatch:\n"
                    f"Source: {record.source_file}\n"
                    f"Expected: {record.row_count:,}\n"
                    f"Actual: {len(index):,}"
                )

            invalid = (
                (index != 0)
                & (index != 1)
                & (index != 2)
            )

            if np.any(invalid):
                invalid_count = int(
                    np.sum(invalid)
                )

                raise ValueError(
                    "Invalid split codes found:\n"
                    f"Source: {record.source_file}\n"
                    f"Invalid values: {invalid_count:,}"
                )

            train_count = int(
                np.sum(index == 0)
            )

            validation_count = int(
                np.sum(index == 1)
            )

            test_count = int(
                np.sum(index == 2)
            )

            if train_count != record.train_rows:
                raise ValueError(
                    "Train count mismatch:\n"
                    f"Source: {record.source_file}\n"
                    f"Summary: {record.train_rows:,}\n"
                    f"Index: {train_count:,}"
                )

            if validation_count != record.validation_rows:
                raise ValueError(
                    "Validation count mismatch:\n"
                    f"Source: {record.source_file}\n"
                    f"Summary: {record.validation_rows:,}\n"
                    f"Index: {validation_count:,}"
                )

            if test_count != record.test_rows:
                raise ValueError(
                    "Test count mismatch:\n"
                    f"Source: {record.source_file}\n"
                    f"Summary: {record.test_rows:,}\n"
                    f"Index: {test_count:,}"
                )

        print(
            "Source split-index validation: PASS"
        )

    # ------------------------------------------------------------------
    # CSV chunk reader
    # ------------------------------------------------------------------

    def iter_chunks(
        self,
        split: str,
        device: Optional[str] = None,
        source_index: Optional[int] = None,
    ) -> Iterator[DataChunk]:

        self._validate_split(
            split
        )

        if (
            device is not None
            and source_index is not None
        ):
            raise ValueError(
                "Specify either device or "
                "source_index, not both."
            )

        requested_code = SPLIT_CODES[
            split
        ]

        records = self.records

        if device is not None:
            records = [
                record
                for record in records
                if self._extract_device(
                    record.source_file
                ) == device
            ]

        if source_index is not None:
            records = [
                record
                for record in records
                if record.source_index
                == source_index
            ]

        if not records:
            raise ValueError(
                "No matching source files found."
            )

        for record in records:

            yield from self._iter_source_chunks(
                record,
                requested_code,
            )

    # ------------------------------------------------------------------
    # Single source-file chunk reader
    # ------------------------------------------------------------------

    def _iter_source_chunks(
        self,
        record: SourceRecord,
        requested_code: int,
    ) -> Iterator[DataChunk]:

        split_index = np.load(
            record.index_file,
            mmap_mode="r",
        )

        csv_row_start = 0

        for chunk in pd.read_csv(
            record.source_file,
            chunksize=self.chunk_size,
        ):

            chunk_row_count = len(chunk)

            csv_row_end = (
                csv_row_start
                + chunk_row_count
            )

            # Dataset row numbers are 1-based.
            #
            # The NumPy split index is 0-based.
            index_slice = split_index[
                csv_row_start:csv_row_end
            ]

            mask = (
                index_slice
                == requested_code
            )

            if not np.any(mask):

                csv_row_start = csv_row_end
                continue

            selected = chunk.loc[
                mask
            ].copy()

            selected_index = np.flatnonzero(
                mask
            )

            row_numbers = (
                csv_row_start
                + selected_index
                + 1
            )

            attack_family = (
                self._extract_attack_family(
                    record.source_file
                )
            )

            binary_label = (
                0
                if attack_family == "benign"
                else 1
            )

            # Remove any accidental metadata columns
            # if they exist in the source CSV.
            feature_columns = [
                column
                for column in selected.columns
                if column not in NON_FEATURE_COLUMNS
            ]

            features = selected[
                feature_columns
            ]

            # Convert feature values to numeric.
            features = features.apply(
                pd.to_numeric,
                errors="coerce",
            )

            # The N-BaIoT source files should contain
            # numerical feature values. We do not silently
            # drop rows here.
            if features.isna().any().any():

                bad_values = int(
                    features.isna()
                    .sum()
                    .sum()
                )

                raise ValueError(
                    "Non-numeric or missing feature "
                    "values detected.\n"
                    f"Source: {record.source_file}\n"
                    f"Rows in chunk: {chunk_row_count:,}\n"
                    f"Invalid values: {bad_values:,}"
                )

            labels = np.full(
                len(features),
                binary_label,
                dtype=np.int8,
            )

            families = np.full(
                len(features),
                attack_family,
                dtype=object,
            )

            yield DataChunk(
                features=features,
                binary_labels=labels,
                attack_families=families,
                source_file=record.source_file,
                row_numbers=row_numbers,
            )

            csv_row_start = csv_row_end

        if csv_row_start != record.row_count:
            raise ValueError(
                "CSV row count does not match "
                "split index length:\n"
                f"Source: {record.source_file}\n"
                f"Expected: {record.row_count:,}\n"
                f"Read: {csv_row_start:,}"
            )

    # ------------------------------------------------------------------
    # Convenience method for a limited sample
    # ------------------------------------------------------------------

    def sample(
        self,
        split: str,
        n_rows: int = 10,
        device: Optional[str] = None,
    ) -> DataChunk:

        self._validate_split(
            split
        )

        if n_rows <= 0:
            raise ValueError(
                "n_rows must be greater than zero."
            )

        collected_features = []
        collected_labels = []
        collected_families = []
        collected_sources = []
        collected_row_numbers = []

        collected = 0

        for chunk in self.iter_chunks(
            split=split,
            device=device,
        ):

            remaining = (
                n_rows
                - collected
            )

            take = min(
                remaining,
                len(chunk.features),
            )

            if take <= 0:
                break

            collected_features.append(
                chunk.features.iloc[
                    :take
                ]
            )

            collected_labels.append(
                chunk.binary_labels[
                    :take
                ]
            )

            collected_families.append(
                chunk.attack_families[
                    :take
                ]
            )

            collected_sources.extend(
                [chunk.source_file] * take
            )

            collected_row_numbers.append(
                chunk.row_numbers[
                    :take
                ]
            )

            collected += take

            if collected >= n_rows:
                break

        if collected == 0:
            raise RuntimeError(
                "No rows were available for the requested sample."
            )

        features = pd.concat(
            collected_features,
            ignore_index=True,
        )

        labels = np.concatenate(
            collected_labels
        )

        families = np.concatenate(
            collected_families
        )

        row_numbers = np.concatenate(
            collected_row_numbers
        )

        return DataChunk(
            features=features,
            binary_labels=labels,
            attack_families=families,
            source_file=Path(
                str(collected_sources[0])
            ),
            row_numbers=row_numbers,
        )