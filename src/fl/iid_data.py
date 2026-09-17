from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from src.data.nbaiot_loader import (
    DataChunk,
    NBaIoTSplitLoader,
)
from src.data.preprocessing import (
    EXPECTED_FEATURE_COUNT,
    FittedStandardScaler,
)
from src.data.torch_data import (
    BatchData,
    transform_chunk,
)


class IIDClientDataLoader:
    """
    Stream training data belonging to one simulated IID client.

    The IID assignment array is indexed by the global sequential
    training-row ordinal produced by:

        NBaIoTSplitLoader.iter_chunks("train")

    Only the selected client's rows are retained from each CSV chunk.
    The complete training dataset is never loaded into memory.
    """

    def __init__(
        self,
        loader: NBaIoTSplitLoader,
        assignment_path: Path,
        num_clients: int = 9,
    ) -> None:
        self.loader = loader
        self.assignment_path = Path(assignment_path)
        self.num_clients = num_clients

        if num_clients <= 0:
            raise ValueError(
                "num_clients must be greater than zero."
            )

        if not self.assignment_path.is_file():
            raise FileNotFoundError(
                "IID assignment file does not exist:\n"
                f"{self.assignment_path}"
            )

        self.assignment = np.load(
            self.assignment_path,
            allow_pickle=False,
        )

        if self.assignment.ndim != 1:
            raise ValueError(
                "IID assignment must be one-dimensional."
            )

        if len(self.assignment) == 0:
            raise ValueError(
                "IID assignment must not be empty."
            )

        if np.any(self.assignment < 0):
            raise ValueError(
                "IID assignment contains unassigned rows."
            )

        if np.any(self.assignment >= num_clients):
            raise ValueError(
                "IID assignment contains an invalid client index."
            )

        self._source_train_offsets = (
            self._build_source_train_offsets()
        )

        total_training_rows = (
            self._source_train_offsets[-1]
            if self._source_train_offsets
            else 0
        )

        if len(self.assignment) != total_training_rows:
            raise ValueError(
                "IID assignment length does not match the "
                "loader's frozen training split.\n"
                f"Assignment rows: {len(self.assignment):,}\n"
                f"Training rows:  {total_training_rows:,}"
            )

    def _build_source_train_offsets(self) -> list[int]:
        """
        Build cumulative global training-row offsets.

        Each source file contributes one contiguous range of global
        training-row ordinals.
        """

        offsets = [0]
        running_total = 0

        for record in self.loader.source_records:
            running_total += record.train_count
            offsets.append(running_total)

        return offsets

    def count_client_rows(
        self,
        client_id: str,
    ) -> int:
        """
        Return the number of training rows assigned to one IID client.
        """

        client_index = self._client_id_to_index(
            client_id
        )

        return int(
            np.sum(
                self.assignment == client_index
            )
        )

    def iter_chunks(
        self,
        client_id: str,
        split: str = "train",
    ) -> Iterator[DataChunk]:
        """
        Stream DataChunk objects belonging to one IID client.

        IID assignment is currently defined only for the training split.
        Validation and test remain global evaluation datasets.
        """

        if split != "train":
            raise ValueError(
                "IID client assignment is defined only for "
                "the training split."
            )

        client_index = self._client_id_to_index(
            client_id
        )

        global_train_offset = 0

        for record in self.loader.source_records:
            if record.train_count == 0:
                continue

            source_train_start = global_train_offset

            source_train_end = (
                source_train_start
                + record.train_count
            )

            source_assignment = self.assignment[
                source_train_start:source_train_end
            ]

            source_local_offset = 0

            for chunk in self.loader._iter_source_chunks(
                record,
                split="train",
            ):
                chunk_size = len(
                    chunk.binary_labels
                )

                local_start = source_local_offset

                local_end = (
                    local_start + chunk_size
                )

                chunk_assignment = source_assignment[
                    local_start:local_end
                ]

                selected_client_rows = (
                    chunk_assignment == client_index
                )

                if np.any(selected_client_rows):
                    selected_mask = (
                        selected_client_rows
                    )

                    yield DataChunk(
                        features=chunk.features.loc[
                            selected_mask
                        ].reset_index(drop=True),
                        binary_labels=chunk.binary_labels.loc[
                            selected_mask
                        ].reset_index(drop=True),
                        attack_families=chunk.attack_families.loc[
                            selected_mask
                        ].reset_index(drop=True),
                        devices=chunk.devices.loc[
                            selected_mask
                        ].reset_index(drop=True),
                        source_file=chunk.source_file,
                        row_numbers=chunk.row_numbers[
                            selected_mask
                        ],
                        split=chunk.split,
                    )

                source_local_offset = local_end

            if source_local_offset != record.train_count:
                raise RuntimeError(
                    "Source training-row count mismatch.\n"
                    f"Source: {record.source_file}\n"
                    f"Expected: {record.train_count:,}\n"
                    f"Consumed: {source_local_offset:,}"
                )

            global_train_offset = source_train_end

    def iter_torch_batches(
        self,
        client_id: str,
        scaler: FittedStandardScaler,
        batch_size: int = 256,
        device: Optional[str] = None,
    ) -> Iterator[BatchData]:
        """
        Stream standardized PyTorch batches for one IID client.

        Data is read and processed incrementally. The complete client
        dataset is never loaded into memory.
        """

        for chunk in self.iter_chunks(
            client_id=client_id,
            split="train",
        ):
            if (
                chunk.features.shape[1]
                != EXPECTED_FEATURE_COUNT
            ):
                raise ValueError(
                    "Unexpected feature count in IID client chunk.\n"
                    f"Expected: {EXPECTED_FEATURE_COUNT}\n"
                    f"Actual:   {chunk.features.shape[1]}"
                )

            yield from transform_chunk(
                scaler=scaler,
                features=chunk.features,
                labels=chunk.binary_labels,
                batch_size=batch_size,
                device=device,
            )

    def _client_id_to_index(
        self,
        client_id: str,
    ) -> int:
        """
        Convert client_1 ... client_9 into zero-based indices.
        """

        prefix = "client_"

        if not client_id.startswith(prefix):
            raise ValueError(
                "IID client IDs must use the format "
                "'client_1', 'client_2', etc."
            )

        client_number_text = client_id[
            len(prefix):
        ]

        try:
            client_number = int(
                client_number_text
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid IID client ID: {client_id!r}"
            ) from exc

        if not 1 <= client_number <= self.num_clients:
            raise ValueError(
                f"Client number must be between 1 and "
                f"{self.num_clients}."
            )

        return client_number - 1