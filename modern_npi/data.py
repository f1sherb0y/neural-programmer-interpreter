from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Iterator

import numpy as np

from modern_npi.constants import ARG_DEPTHS
from modern_npi.traces import Episode


DIGIT_DEPTH = 10
OBSERVATION_SIZE = 4 * DIGIT_DEPTH + 1
ARGUMENT_SIZE = sum(ARG_DEPTHS)
FEATURE_SIZE = OBSERVATION_SIZE + ARGUMENT_SIZE


@dataclass(frozen=True)
class EncodedEpisode:
    features: np.ndarray
    programs: np.ndarray
    target_end: np.ndarray
    target_program: np.ndarray
    target_arguments: tuple[np.ndarray, np.ndarray, np.ndarray]
    child_mask: np.ndarray


def encode_feature(observation, arguments: tuple[int, int, int]) -> np.ndarray:
    row = np.zeros(FEATURE_SIZE, dtype=np.float32)
    offset = 0
    for digit in observation.digits:
        row[offset + digit] = 1.0
        offset += DIGIT_DEPTH
    row[offset] = observation.at_most_significant_input
    offset += 1
    for argument, depth in zip(arguments, ARG_DEPTHS, strict=True):
        row[offset + int(argument)] = 1.0
        offset += depth
    return row


def encode_features(episode: Episode) -> np.ndarray:
    return np.stack(
        [encode_feature(decision.observation, episode.arguments) for decision in episode.decisions]
    )


def encode_episode(episode: Episode) -> EncodedEpisode:
    decisions = episode.decisions
    arguments = tuple(
        np.asarray([int(step.next_arguments[index]) for step in decisions], dtype=np.int32)
        for index in range(3)
    )
    return EncodedEpisode(
        features=encode_features(episode),
        programs=np.full(len(decisions), int(episode.program), dtype=np.int32),
        target_end=np.asarray([step.end_after_call for step in decisions], dtype=np.int32),
        target_program=np.asarray([int(step.next_program) for step in decisions], dtype=np.int32),
        target_arguments=arguments,
        child_mask=np.asarray([step.has_child for step in decisions], dtype=np.float32),
    )


def select_episodes(
    episodes: list[Episode],
    *,
    maximum_act_episodes: int | None = None,
    seed: int = 1,
) -> list[Episode]:
    if maximum_act_episodes is None:
        return episodes
    act = [episode for episode in episodes if episode.program.name == "ACT"]
    other = [episode for episode in episodes if episode.program.name != "ACT"]
    random.Random(seed).shuffle(act)
    return other + act[:maximum_act_episodes]


class EpisodeBatches:
    """Batches equal-length invocations, avoiding padding and loss masks."""

    def __init__(self, episodes: list[Episode]):
        grouped: dict[int, list[EncodedEpisode]] = defaultdict(list)
        for episode in episodes:
            grouped[len(episode.decisions)].append(encode_episode(episode))
        self.groups = dict(grouped)
        self.size = sum(len(group) for group in self.groups.values())

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
                yield (
                    np.stack([record.features for record in batch]),
                    np.stack([record.programs for record in batch]),
                    np.stack([record.target_end for record in batch]),
                    np.stack([record.target_program for record in batch]),
                    np.stack([record.target_arguments[0] for record in batch]),
                    np.stack([record.target_arguments[1] for record in batch]),
                    np.stack([record.target_arguments[2] for record in batch]),
                    np.stack([record.child_mask for record in batch]),
                )
