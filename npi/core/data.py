import random
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from npi.core.spec import ObservationCodec
from npi.core.traces import Episode

DEFAULT_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


@dataclass(frozen=True)
class EncodedEpisode:
    features: np.ndarray
    program: int
    target_end: np.ndarray
    target_program: np.ndarray
    target_arguments: tuple[np.ndarray, ...]
    child_mask: np.ndarray

    @property
    def length(self) -> int:
        return self.features.shape[0]


@dataclass(frozen=True)
class TrainingBatch:
    features: np.ndarray
    programs: np.ndarray
    target_end: np.ndarray
    target_program: np.ndarray
    target_arguments: tuple[np.ndarray, ...]
    child_mask: np.ndarray
    sequence_mask: np.ndarray


def encode_episode(episode: Episode, codec: ObservationCodec) -> EncodedEpisode:
    decisions = episode.decisions
    if not decisions:
        raise ValueError("An invocation episode cannot be empty")
    argument_count = len(codec.spec.argument_depths)
    return EncodedEpisode(
        features=np.stack(
            [
                codec.encode_feature(decision.observation, episode.arguments)
                for decision in decisions
            ]
        ),
        program=int(episode.program),
        target_end=np.asarray([decision.end for decision in decisions], np.int32),
        target_program=np.asarray(
            [decision.next_program for decision in decisions], np.int32
        ),
        target_arguments=tuple(
            np.asarray(
                [decision.next_arguments[index] for decision in decisions],
                np.int32,
            )
            for index in range(argument_count)
        ),
        child_mask=np.asarray(
            [decision.has_child for decision in decisions], np.float32
        ),
    )


class EpisodeDataset:
    """Padded length buckets shared by every task.

    Exact-length grouping created many tiny GPU batches. Power-of-two buckets
    trade modest padding for substantially larger and shape-stable XLA batches.
    """

    def __init__(
        self,
        episodes: list[Episode],
        codec: ObservationCodec,
        bucket_boundaries: tuple[int, ...] = DEFAULT_BUCKETS,
    ):
        self.spec = codec.spec
        self.records = [encode_episode(episode, codec) for episode in episodes]
        if not self.records:
            raise ValueError("Dataset requires at least one episode")
        self.bucket_boundaries = bucket_boundaries
        self.size = len(self.records)
        self.decisions = sum(record.length for record in self.records)

    def _bucket(self, length: int) -> int:
        for boundary in self.bucket_boundaries:
            if length <= boundary:
                return boundary
        boundary = self.bucket_boundaries[-1]
        while boundary < length:
            boundary *= 2
        return boundary

    def batches(
        self,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ) -> Iterator[TrainingBatch]:
        groups: dict[int, list[EncodedEpisode]] = {}
        for record in self.records:
            groups.setdefault(self._bucket(record.length), []).append(record)
        rng = random.Random(seed)
        boundaries = list(groups)
        if shuffle:
            rng.shuffle(boundaries)
        for boundary in boundaries:
            records = list(groups[boundary])
            if shuffle:
                rng.shuffle(records)
            for start in range(0, len(records), batch_size):
                yield self._pad(records[start : start + batch_size], boundary)

    def _pad(self, records: list[EncodedEpisode], length: int) -> TrainingBatch:
        batch = len(records)
        features = np.zeros((batch, length, self.spec.feature_size), np.float32)
        programs = np.zeros((batch, length), np.int32)
        target_end = np.zeros((batch, length), np.int32)
        target_program = np.zeros((batch, length), np.int32)
        target_arguments = tuple(
            np.zeros((batch, length), np.int32) for _ in self.spec.argument_depths
        )
        child_mask = np.zeros((batch, length), np.float32)
        sequence_mask = np.zeros((batch, length), np.float32)
        for row, record in enumerate(records):
            count = record.length
            features[row, :count] = record.features
            programs[row, :count] = record.program
            target_end[row, :count] = record.target_end
            target_program[row, :count] = record.target_program
            for destination, source in zip(
                target_arguments, record.target_arguments, strict=True
            ):
                destination[row, :count] = source
            child_mask[row, :count] = record.child_mask
            sequence_mask[row, :count] = 1.0
        return TrainingBatch(
            features,
            programs,
            target_end,
            target_program,
            target_arguments,
            child_mask,
            sequence_mask,
        )
