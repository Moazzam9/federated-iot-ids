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
    "attack": 1,
}

EXPECTED_DEVICE_NAMES = {
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
}

EXPECTED_FEATURE_COUNT = 115


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SourceRecord:
    """
    Metadata describing one audited N-BaIoT source CSV.
    """

    source_file: Path
    index_file: Path
    device: str
    attack_family: str
    row_count: int
    train_count: int
    validation_count: int
    test_count: int


@dataclass
class DataChunk:
    """
    Data returned by the split-aware loader.

    `features` contains the 115 original N-BaIoT numerical
    feature columns.

    The remaining fields are metadata and labels supplied by
    the loader.
    """

    features: pd.DataFrame
    binary_labels: pd.Series
    attack_families: pd.Series
    devices: pd.Series
    source_file: str | list[str]
    row_numbers: np.ndarray
    split: str

    # ------------------------------------------------------------------
    # Backward-compatible aliases
    # ------------------------------------------------------------------

    @property
    def X(self) -> pd.DataFrame:
        return self.features

    @property
    def y(self) -> pd.Series:
        return self.binary_labels

    @property
    def device(self) -> str:
        unique_devices = self.devices.drop_duplicates().tolist()

        if len(unique_devices) == 1:
            return str(unique_devices[0])

        return "multiple"

    @property
    def attack_family(self) -> str:
        unique_families = (
            self.attack_families
            .drop_duplicates()
            .tolist()
        )

        if len(unique_families) == 1:
            return str(unique_families[0])

        return "multiple"


# ----------------------------------------------------------------------
# N-BaIoT split-aware loader
# ----------------------------------------------------------------------


class NBaIoTSplitLoader:
    """
    Split-aware data-access layer for the audited N-BaIoT dataset.

    Important:
        This loader does NOT create new train/validation/test splits.

    It reads the previously generated source-level split indexes:

        0 = train
        1 = validation
        2 = test

    The source-index summary contains:

        source_index
        source_file
        index_file
        row_count
        train_rows
        validation_rows
        test_rows

    The raw N-BaIoT CSV files contain 115 numerical feature columns.
    They do not contain a label column.

    Binary labels and metadata are therefore derived from the audited
    source-file information.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        data_root: Optional[Path] = None,
        split_root: Optional[Path] = None,
        chunk_size: int = 50_000,
    ) -> None:

        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )

        self.data_root = (
            Path(data_root)
            if data_root is not None
            else self.project_root.parent / "federated-iot-temp"
        )

        self.split_root = (
            Path(split_root).resolve()
            if split_root is not None
            else (
                self.project_root
                / "data"
                / "processed"
                / "splits"
                / "source_indexes"
            )
        )

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        self.chunksize = int(chunk_size)

        # Locate the actual N-BaIoT dataset directory.
        self.data_root = self._resolve_dataset_root(
            self.data_root
        )

        self.source_summary_path = (
            self.project_root
            / "data"
            / "processed"
            / "splits"
            / "source_index_summary.csv"
        )

        self.source_records = self._load_source_summary()

        self._validate_source_summary_totals()

    # ------------------------------------------------------------------
    # Dataset root discovery
    # ------------------------------------------------------------------

    def _resolve_dataset_root(
        self,
        data_root: Path,
    ) -> Path:
        """
        Locate the actual N-BaIoT dataset root.

        Supported layouts:

            data_root/
                <nine device directories>

        or:

            data_root/
                <dataset directory>/
                    <nine device directories>
        """

        data_root = Path(data_root).resolve()

        if not data_root.is_dir():
            raise FileNotFoundError(
                "The specified N-BaIoT data root does not exist:\n"
                f"{data_root}"
            )

        expected_devices = EXPECTED_DEVICE_NAMES

        # Candidate 1: data_root itself contains all devices.
        direct_devices = {
            path.name
            for path in data_root.iterdir()
            if path.is_dir()
        }

        if expected_devices.issubset(direct_devices):
            return data_root

        # Candidate 2: an immediate child contains all devices.
        for child in sorted(data_root.iterdir()):
            if not child.is_dir():
                continue

            child_devices = {
                path.name
                for path in child.iterdir()
                if path.is_dir()
            }

            if expected_devices.issubset(child_devices):
                return child

        raise FileNotFoundError(
            "Could not locate the N-BaIoT dataset root.\n"
            f"Checked: {data_root}\n\n"
            "Expected to find these nine device directories:\n"
            + "\n".join(
                f"  - {device}"
                for device in sorted(expected_devices)
            )
        )

    # ------------------------------------------------------------------
    # Source-file resolution
    # ------------------------------------------------------------------

    def _resolve_source_file(
        self,
        source_file: str | Path,
    ) -> Path:
        """
        Resolve a stable audit source path to the physical CSV.

        Stable benign path:

            Device\\benign_traffic.csv

        Stable attack path:

            Device\\gafgyt_attacks\\combo.csv

        Physical attack path:

            attack_extract\\Device\\gafgyt_attacks\\combo.csv
        """

        source_file = Path(source_file)

        if source_file.is_absolute():
            if source_file.is_file():
                return source_file.resolve()

            raise FileNotFoundError(
                "Source CSV does not exist:\n"
                f"{source_file}"
            )

        # Direct benign source.
        direct_candidate = (
            self.data_root / source_file
        ).resolve()

        if direct_candidate.is_file():
            return direct_candidate

        # Attack source.
        attack_candidate = (
            self.data_root
            / "attack_extract"
            / source_file
        ).resolve()

        if attack_candidate.is_file():
            return attack_candidate

        # Recursive fallback.
        attack_root = (
            self.data_root / "attack_extract"
        )

        if attack_root.is_dir():

            matches = list(
                attack_root.glob(
                    f"**/{source_file.name}"
                )
            )

            matching_paths = []

            target_parts = tuple(
                part.lower()
                for part in source_file.parts
            )

            for match in matches:

                if not match.is_file():
                    continue

                try:
                    relative = match.relative_to(
                        attack_root
                    )
                except ValueError:
                    continue

                relative_parts = tuple(
                    part.lower()
                    for part in relative.parts
                )

                if relative_parts == target_parts:
                    matching_paths.append(match)

            if len(matching_paths) == 1:
                return matching_paths[0].resolve()

            if len(matching_paths) > 1:
                raise RuntimeError(
                    "Multiple physical files matched "
                    "stable source path:\n"
                    f"{source_file}\n\n"
                    + "\n".join(
                        f"  - {match}"
                        for match in matching_paths
                    )
                )

        raise FileNotFoundError(
            "Source CSV does not exist.\n"
            f"Stable source path: {source_file}\n"
            f"Dataset root: {self.data_root}\n\n"
            "Checked:\n"
            f"  - {direct_candidate}\n"
            f"  - {attack_candidate}"
        )

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_device(
        source_file: Path,
    ) -> str:
        """
        Extract the logical N-BaIoT device.
        """

        parts = list(source_file.parts)

        if (
            parts
            and parts[0] in EXPECTED_DEVICE_NAMES
        ):
            return parts[0]

        if "attack_extract" in parts:

            attack_index = parts.index(
                "attack_extract"
            )

            if attack_index + 1 < len(parts):

                candidate = parts[
                    attack_index + 1
                ]

                if candidate in EXPECTED_DEVICE_NAMES:
                    return candidate

        for part in parts:
            if part in EXPECTED_DEVICE_NAMES:
                return part

        raise ValueError(
            "Could not determine device from source path:\n"
            f"{source_file}"
        )

    @staticmethod
    def _extract_attack_family(
        source_file: Path,
    ) -> str:
        """
        Extract attack family from the stable source path.

        Returns one of:

            benign
            gafgyt
            mirai
        """

        parts_lower = {
            part.lower()
            for part in source_file.parts
        }

        if "benign_traffic.csv" in parts_lower:
            return "benign"

        if "gafgyt_attacks" in parts_lower:
            return "gafgyt"

        if "mirai_attacks" in parts_lower:
            return "mirai"

        raise ValueError(
            "Could not determine attack family from source path:\n"
            f"{source_file}"
        )

    # ------------------------------------------------------------------
    # Source summary
    # ------------------------------------------------------------------

    def _load_source_summary(
        self,
    ) -> list[SourceRecord]:
        """
        Load source-level split metadata.
        """

        if not self.source_summary_path.is_file():
            raise FileNotFoundError(
                "Source index summary does not exist:\n"
                f"{self.source_summary_path}\n\n"
                "Run build_split_indexes.py first."
            )

        summary = pd.read_csv(
            self.source_summary_path
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
                "Source index summary is missing "
                "required columns:\n"
                + "\n".join(
                    f"  - {column}"
                    for column in sorted(missing)
                )
            )

        records: list[SourceRecord] = []

        for row in summary.itertuples(
            index=False
        ):

            stable_source_file = Path(
                str(row.source_file)
            )

            actual_source_file = (
                self._resolve_source_file(
                    stable_source_file
                )
            )

            device = (
                self._extract_device(
                    stable_source_file
                )
            )

            attack_family = (
                self._extract_attack_family(
                    stable_source_file
                )
            )

            records.append(
                SourceRecord(
                    source_file=actual_source_file,
                    index_file=Path(
                        str(row.index_file)
                    ),
                    device=device,
                    attack_family=attack_family,
                    row_count=int(
                        row.row_count
                    ),
                    train_count=int(
                        row.train_rows
                    ),
                    validation_count=int(
                        row.validation_rows
                    ),
                    test_count=int(
                        row.test_rows
                    ),
                )
            )

        if not records:
            raise ValueError(
                "Source index summary contains no source records."
            )

        return records

    # ------------------------------------------------------------------
    # Summary-level validation
    # ------------------------------------------------------------------

    def _validate_source_summary_totals(
        self,
    ) -> None:
        """
        Validate basic arithmetic consistency of the source summary.

        This does not replace full split-index validation.
        """

        for record in self.source_records:

            split_total = (
                record.train_count
                + record.validation_count
                + record.test_count
            )

            if split_total != record.row_count:
                raise ValueError(
                    "Source summary split counts do not sum "
                    "to source row count.\n"
                    f"Source: {record.source_file}\n"
                    f"Rows: {record.row_count}\n"
                    f"Train: {record.train_count}\n"
                    f"Validation: {record.validation_count}\n"
                    f"Test: {record.test_count}"
                )

    # ------------------------------------------------------------------
    # Public dataset information
    # ------------------------------------------------------------------

    def source_count(self) -> int:
        """
        Return the number of audited source CSV files.
        """

        return len(self.source_records)

    def total_rows(self) -> int:
        """
        Return the total number of rows across all source CSV files.
        """

        return sum(
            record.row_count
            for record in self.source_records
        )

    def total_rows_for_split(
        self,
        split: str,
    ) -> int:
        """
        Return the total number of rows assigned to a split.
        """

        if split not in SPLIT_CODES:
            raise ValueError(
                "Invalid split.\n"
                f"Received: {split}\n"
                f"Expected one of: "
                f"{sorted(SPLIT_CODES)}"
            )

        if split == "train":
            return sum(
                record.train_count
                for record in self.source_records
            )

        if split == "validation":
            return sum(
                record.validation_count
                for record in self.source_records
            )

        return sum(
            record.test_count
            for record in self.source_records
        )

    def list_devices(self) -> list[str]:
        """
        Return the logical N-BaIoT device names in stable order.
        """

        devices = {
            record.device
            for record in self.source_records
        }

        unexpected = devices.difference(
            EXPECTED_DEVICE_NAMES
        )

        if unexpected:
            raise ValueError(
                "Unexpected device names found:\n"
                + "\n".join(
                    f"  - {device}"
                    for device in sorted(unexpected)
                )
            )

        return sorted(devices)

    # ------------------------------------------------------------------
    # Split index path
    # ------------------------------------------------------------------

    def _get_index_path(
        self,
        record: SourceRecord,
    ) -> Path:
        """
        Return the physical .npy split-index path.
        """

        index_path = (
            self.split_root
            / record.index_file
        )

        if not index_path.is_file():
            raise FileNotFoundError(
                "Split index does not exist:\n"
                f"{index_path}\n\n"
                f"Source CSV: {record.source_file}"
            )

        return index_path

    # ------------------------------------------------------------------
    # Split index loading
    # ------------------------------------------------------------------

    def _load_split_codes(
        self,
        record: SourceRecord,
    ) -> np.ndarray:
        """
        Load the split-code array for one source CSV.
        """

        index_path = (
            self._get_index_path(
                record
            )
        )

        split_codes = np.load(
            index_path,
            allow_pickle=False,
        )

        if split_codes.ndim != 1:
            raise ValueError(
                "Split index must be one-dimensional.\n"
                f"Index: {index_path}\n"
                f"Shape: {split_codes.shape}"
            )

        return split_codes

    # ------------------------------------------------------------------
    # Split index validation
    # ------------------------------------------------------------------

    def validate_source_indexes(
        self,
    ) -> None:
        """
        Validate every source-level split index.
        """

        for record in self.source_records:

            if not record.source_file.is_file():
                raise FileNotFoundError(
                    "Source CSV does not exist:\n"
                    f"{record.source_file}"
                )

            split_codes = (
                self._load_split_codes(
                    record
                )
            )

            if len(split_codes) != record.row_count:
                raise ValueError(
                    "Split index length does not match "
                    "source row count.\n"
                    f"Source: {record.source_file}\n"
                    f"Expected: {record.row_count}\n"
                    f"Actual: {len(split_codes)}"
                )

            unique_codes = set(
                np.unique(
                    split_codes
                ).tolist()
            )

            invalid_codes = (
                unique_codes.difference(
                    SPLIT_CODES.values()
                )
            )

            if invalid_codes:
                raise ValueError(
                    "Invalid split code(s) found.\n"
                    f"Source: {record.source_file}\n"
                    f"Invalid codes: "
                    f"{sorted(invalid_codes)}"
                )

            train_count = int(
                np.count_nonzero(
                    split_codes
                    == SPLIT_CODES["train"]
                )
            )

            validation_count = int(
                np.count_nonzero(
                    split_codes
                    == SPLIT_CODES["validation"]
                )
            )

            test_count = int(
                np.count_nonzero(
                    split_codes
                    == SPLIT_CODES["test"]
                )
            )

            if train_count != record.train_count:
                raise ValueError(
                    "Train count mismatch.\n"
                    f"Source: {record.source_file}\n"
                    f"Expected: {record.train_count}\n"
                    f"Actual: {train_count}"
                )

            if validation_count != record.validation_count:
                raise ValueError(
                    "Validation count mismatch.\n"
                    f"Source: {record.source_file}\n"
                    f"Expected: {record.validation_count}\n"
                    f"Actual: {validation_count}"
                )

            if test_count != record.test_count:
                raise ValueError(
                    "Test count mismatch.\n"
                    f"Source: {record.source_file}\n"
                    f"Expected: {record.test_count}\n"
                    f"Actual: {test_count}"
                )

    # ------------------------------------------------------------------
    # Feature validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_features(
        features: pd.DataFrame,
        source_file: Path,
    ) -> None:
        """
        Validate the raw N-BaIoT feature matrix.
        """

        if len(features.columns) != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                "Unexpected N-BaIoT feature count.\n"
                f"Source: {source_file}\n"
                f"Expected: {EXPECTED_FEATURE_COUNT}\n"
                f"Actual: {len(features.columns)}"
            )

        for column in features.columns:

            features[column] = pd.to_numeric(
                features[column],
                errors="coerce",
            )

        if features.isna().any().any():

            nan_columns = (
                features.columns[
                    features.isna().any()
                ].tolist()
            )

            raise ValueError(
                "NaN values detected after numeric conversion.\n"
                f"Source: {source_file}\n"
                f"Columns: {nan_columns}"
            )

    # ------------------------------------------------------------------
    # Chunk iteration
    # ------------------------------------------------------------------

    def iter_chunks(
        self,
        split: str,
        devices: Optional[list[str]] = None,
        attack_families: Optional[list[str]] = None,
    ) -> Iterator[DataChunk]:
        """
        Iterate through a selected split.

        Optional filtering can be performed by logical device and/or
        attack family.
        """

        if split not in SPLIT_CODES:
            raise ValueError(
                "Invalid split.\n"
                f"Received: {split}\n"
                f"Expected one of: "
                f"{sorted(SPLIT_CODES)}"
            )

        selected_devices = (
            set(devices)
            if devices is not None
            else None
        )

        selected_families = (
            set(attack_families)
            if attack_families is not None
            else None
        )

        if selected_devices is not None:

            invalid_devices = (
                selected_devices
                - EXPECTED_DEVICE_NAMES
            )

            if invalid_devices:
                raise ValueError(
                    "Unknown device(s):\n"
                    + "\n".join(
                        f"  - {device}"
                        for device in sorted(
                            invalid_devices
                        )
                    )
                )

        if selected_families is not None:

            valid_families = {
                "benign",
                "gafgyt",
                "mirai",
            }

            invalid_families = (
                selected_families
                - valid_families
            )

            if invalid_families:
                raise ValueError(
                    "Unknown attack family/families:\n"
                    + "\n".join(
                        f"  - {family}"
                        for family in sorted(
                            invalid_families
                        )
                    )
                )

        for record in self.source_records:

            if (
                selected_devices is not None
                and record.device
                not in selected_devices
            ):
                continue

            if (
                selected_families is not None
                and record.attack_family
                not in selected_families
            ):
                continue

            yield from self._iter_source_chunks(
                record=record,
                split=split,
            )

    # ------------------------------------------------------------------
    # Source chunk iterator
    # ------------------------------------------------------------------

    def _iter_source_chunks(
        self,
        record: SourceRecord,
        split: str,
    ) -> Iterator[DataChunk]:
        """
        Read one source CSV in chunks and retain only rows belonging
        to the requested split.
        """

        split_code = SPLIT_CODES[split]

        split_codes = self._load_split_codes(
            record
        )

        row_start = 0

        for chunk in pd.read_csv(
            record.source_file,
            chunksize=self.chunksize,
        ):

            row_end = (
                row_start + len(chunk)
            )

            chunk_split_codes = (
                split_codes[
                    row_start:row_end
                ]
            )

            mask = (
                chunk_split_codes
                == split_code
            )

            if not np.any(mask):
                row_start = row_end
                continue

            selected = chunk.loc[
                mask
            ].copy()

            original_row_numbers = (
                np.arange(
                    row_start,
                    row_end,
                    dtype=np.int64,
                )[mask]
            )

            # ----------------------------------------------------------
            # The raw N-BaIoT CSV contains exactly 115 numerical
            # feature columns and no label column.
            # ----------------------------------------------------------

            features = selected.copy()

            self._validate_features(
                features=features,
                source_file=record.source_file,
            )

            features.reset_index(
                drop=True,
                inplace=True,
            )

            # ----------------------------------------------------------
            # Binary label
            # ----------------------------------------------------------

            binary_label = BINARY_LABEL_MAP.get(
                "benign"
                if record.attack_family == "benign"
                else "attack"
            )

            binary_labels = pd.Series(
                np.full(
                    len(selected),
                    binary_label,
                    dtype=np.int8,
                ),
                name="binary_label",
            )

            # ----------------------------------------------------------
            # Metadata
            # ----------------------------------------------------------

            attack_families = pd.Series(
                [record.attack_family]
                * len(selected),
                name="attack_family",
            )

            devices = pd.Series(
                [record.device]
                * len(selected),
                name="device",
            )

            source_file = [
                str(record.source_file)
            ] * len(selected)

            row_numbers = np.asarray(
                original_row_numbers,
                dtype=np.int64,
            )

            yield DataChunk(
                features=features,
                binary_labels=binary_labels,
                attack_families=attack_families,
                devices=devices,
                source_file=source_file,
                row_numbers=row_numbers,
                split=split,
            )

            row_start = row_end

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        split: str,
        n_rows: int = 1_000,
        devices: Optional[list[str]] = None,
        attack_families: Optional[list[str]] = None,
        device: Optional[str] = None,
    ) -> DataChunk:
        """
        Return up to n_rows from the requested split.

        `device` is supported as a convenience for the verification
        script. It is equivalent to supplying a single-item `devices`
        list.
        """

        if n_rows <= 0:
            raise ValueError(
                "n_rows must be greater than zero."
            )

        if device is not None:

            if devices is not None:
                raise ValueError(
                    "Use either 'device' or 'devices', "
                    "not both."
                )

            devices = [device]

        collected_features: list[pd.DataFrame] = []
        collected_labels: list[pd.Series] = []
        collected_families: list[pd.Series] = []
        collected_devices: list[pd.Series] = []
        collected_sources: list[str] = []
        collected_row_numbers: list[np.ndarray] = []

        collected_rows = 0

        for chunk in self.iter_chunks(
            split=split,
            devices=devices,
            attack_families=attack_families,
        ):

            remaining = (
                n_rows - collected_rows
            )

            take = min(
                remaining,
                len(chunk.features),
            )

            collected_features.append(
                chunk.features.iloc[:take].copy()
            )

            collected_labels.append(
                chunk.binary_labels.iloc[:take].copy()
            )

            collected_families.append(
                chunk.attack_families.iloc[:take].copy()
            )

            collected_devices.append(
                chunk.devices.iloc[:take].copy()
            )

            collected_sources.extend(
                chunk.source_file[:take]
            )

            collected_row_numbers.append(
                chunk.row_numbers[:take].copy()
            )

            collected_rows += take

            if collected_rows >= n_rows:
                break

        if not collected_features:
            raise ValueError(
                "No rows were available for the requested sample."
            )

        features = pd.concat(
            collected_features,
            ignore_index=True,
        )

        binary_labels = pd.concat(
            collected_labels,
            ignore_index=True,
        )

        attack_families = pd.concat(
            collected_families,
            ignore_index=True,
        )

        sampled_devices = pd.concat(
            collected_devices,
            ignore_index=True,
        )

        row_numbers = np.concatenate(
            collected_row_numbers
        )

        if len(features) != len(binary_labels):
            raise RuntimeError(
                "Sample feature/label length mismatch."
            )

        return DataChunk(
            features=features,
            binary_labels=binary_labels,
            attack_families=attack_families,
            devices=sampled_devices,
            source_file=(
                collected_sources[0]
                if len(set(collected_sources)) == 1
                else collected_sources
            ),
            row_numbers=row_numbers,
            split=split,
        )