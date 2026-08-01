from npi.core.data import EpisodeDataset
from npi.tasks.addition.codec import CODEC
from npi.tasks.addition.traces import training_traces


def make_dataset(examples_per_length: int, maximum_length: int, seed: int):
    traces = training_traces(examples_per_length, maximum_length, seed)
    return EpisodeDataset(
        [episode for trace in traces for episode in trace.episodes],
        CODEC,
    )
