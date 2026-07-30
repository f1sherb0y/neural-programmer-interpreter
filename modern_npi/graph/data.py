from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Iterator

import numpy as np

from modern_npi.graph.constants import ARGUMENT_DEPTHS
from modern_npi.graph.traces import Episode


OBSERVATION_SIZE = 14
ARGUMENT_SIZE = sum(ARGUMENT_DEPTHS)
FEATURE_SIZE = OBSERVATION_SIZE + ARGUMENT_SIZE


@dataclass(frozen=True)
class EncodedEpisode:
    features: np.ndarray
    programs: np.ndarray
    target_end: np.ndarray
    target_program: np.ndarray
    target_arguments: tuple[np.ndarray, ...]
    child_mask: np.ndarray


def encode_feature(observation, arguments: tuple[int, ...]) -> np.ndarray:
    row = np.zeros(FEATURE_SIZE, dtype=np.float32)
    observed = observation.as_tuple()
    row[: len(observed)] = observed
    offset = OBSERVATION_SIZE
    for argument, depth in zip(arguments, ARGUMENT_DEPTHS, strict=True):
        row[offset + int(argument)] = 1.0
        offset += depth
    return row


def encode_episode(episode: Episode) -> EncodedEpisode:
    decisions = episode.decisions
    target_arguments = tuple(
        np.asarray(
            [int(decision.next_arguments[index]) for decision in decisions],
            dtype=np.int64,
        )
        for index in range(len(ARGUMENT_DEPTHS))
    )
    return EncodedEpisode(
        features=np.stack(
            [encode_feature(decision.observation, episode.arguments) for decision in decisions]
        ),
        programs=np.full(len(decisions), int(episode.program), dtype=np.int64),
        target_end=np.asarray([decision.end for decision in decisions], dtype=np.int64),
        target_program=np.asarray(
            [int(decision.next_program) for decision in decisions], dtype=np.int64
        ),
        target_arguments=target_arguments,
        child_mask=np.asarray(
            [decision.has_child for decision in decisions], dtype=np.float32
        ),
    )


class EpisodeBatches:
    def __init__(self, episodes: list[Episode]):
        grouped: dict[int, list[EncodedEpisode]] = defaultdict(list)
        for episode in episodes:
            grouped[len(episode.decisions)].append(encode_episode(episode))
        self.groups = dict(grouped)
        self.size = sum(len(group) for group in self.groups.values())
        self.decisions = sum(
            length * len(group) for length, group in self.groups.items()
        )

    def batches(
        self,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ) -> Iterator[tuple[np.ndarray, ...]]:
        rng = random.Random(seed)
        lengths = list(self.groups)
        if shuffle:
            rng.shuffle(lengths)
        for length in lengths:
            records = list(self.groups[length])
            if shuffle:
                rng.shuffle(records)
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                argument_batches = [
                    np.stack([record.target_arguments[index] for record in batch])
                    for index in range(len(ARGUMENT_DEPTHS))
                ]
                yield (
                    np.stack([record.features for record in batch]),
                    np.stack([record.programs for record in batch]),
                    np.stack([record.target_end for record in batch]),
                    np.stack([record.target_program for record in batch]),
                    *argument_batches,
                    np.stack([record.child_mask for record in batch]),
                )
