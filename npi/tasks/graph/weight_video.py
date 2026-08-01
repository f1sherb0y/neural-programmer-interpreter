import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf

from npi.core.experiment import set_seed
from npi.core.hardware import configure_tensorflow
from npi.core.model import NeuralProgrammerInterpreter
from npi.core.trainer import Trainer
from npi.tasks.graph.data import make_dataset
from npi.tasks.graph.spec import SPEC

DEFAULT_OUTPUT = Path("artifacts/graph_random_weight_evolution.mp4")


def exact_raster_shape(parameter_count: int) -> tuple[int, int]:
    width = math.isqrt(parameter_count)
    while parameter_count % width:
        width -= 1
    return parameter_count // width, width


def parameter_metadata(model, width: int):
    variables = []
    offset = 0
    for variable in model.trainable_variables:
        size = int(np.prod(variable.shape))
        variables.append(
            {
                "name": getattr(variable, "path", variable.name),
                "shape": list(variable.shape),
                "size": size,
                "offset": offset,
                "first_pixel": [offset // width, offset % width],
                "last_pixel": [
                    (offset + size - 1) // width,
                    (offset + size - 1) % width,
                ],
            }
        )
        offset += size
    return variables


class WeightVideoWriter:
    def __init__(
        self,
        path: Path,
        parameter_count: int,
        *,
        frame_rate: int,
        magnitude_maximum: float,
        colormap: str,
    ):
        self.path = path
        self.width, self.height = exact_raster_shape(parameter_count)
        self.frame_rate = frame_rate
        self.magnitude_maximum = magnitude_maximum
        self.denominator = np.log1p(magnitude_maximum / 1e-4)
        colors = matplotlib.colormaps[colormap](np.linspace(0.0, 1.0, 4096))[:, :3]
        self.palette = np.asarray(np.round(colors * 255.0), dtype=np.uint8)
        self.parameters = np.empty(parameter_count, dtype=np.float32)
        self.frame_count = 0
        self.observed_maximum = 0.0
        self.clipped_values = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(frame_rate),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv444p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def add(self, model) -> None:
        offset = 0
        for variable in model.trainable_variables:
            values = variable.numpy().reshape(-1)
            self.parameters[offset : offset + values.size] = values
            offset += values.size
        magnitudes = np.abs(self.parameters)
        self.observed_maximum = max(self.observed_maximum, float(magnitudes.max()))
        self.clipped_values += int(
            np.count_nonzero(magnitudes > self.magnitude_maximum)
        )
        normalized = np.log1p(magnitudes / 1e-4) / self.denominator
        indices = np.asarray(np.clip(normalized, 0.0, 1.0) * 4095, dtype=np.uint16)
        frame = self.palette[indices]
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as error:
            details = self.process.stderr.read().decode(errors="replace")
            raise RuntimeError(f"ffmpeg stopped while encoding: {details}") from error
        self.frame_count += 1

    def close(self) -> None:
        self.process.stdin.close()
        return_code = self.process.wait()
        details = self.process.stderr.read().decode(errors="replace")
        if return_code:
            raise RuntimeError(f"ffmpeg exited with status {return_code}: {details}")


def parser():
    result = argparse.ArgumentParser(
        description="Train graph NPI and render one weight-map frame per interval"
    )
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--maximum-train-nodes", type=int, default=30)
    result.add_argument("--training-examples-per-size", type=int, default=32)
    result.add_argument("--steps", type=int, default=120_000)
    result.add_argument("--frame-interval", type=int, default=50)
    result.add_argument("--frame-rate", type=int, default=30)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--magnitude-maximum", type=float, default=4.0)
    result.add_argument("--colormap", default="magma")
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--cpu", action="store_true")
    result.add_argument("--no-xla", action="store_true")
    result.add_argument("--reference-checkpoint", type=Path)
    return result


def main():
    args = parser().parse_args()
    if args.steps < 1 or args.steps % args.frame_interval:
        raise ValueError("Steps must be positive and divisible by the frame interval")
    if args.frame_interval < 1 or args.frame_rate < 1:
        raise ValueError("Frame interval and frame rate must be positive")
    if args.magnitude_maximum <= 0:
        raise ValueError("Magnitude maximum must be positive")
    configure_tensorflow(device="cpu" if args.cpu else "auto")
    set_seed(args.seed)
    training_data, _ = make_dataset(
        2,
        args.maximum_train_nodes,
        args.training_examples_per_size,
        args.seed,
    )
    model = NeuralProgrammerInterpreter(SPEC)
    model.build_for_task()
    trainer = Trainer(
        model,
        args.learning_rate,
        weight_decay=args.weight_decay,
        use_xla=not args.no_xla,
    )
    parameter_count = sum(
        int(np.prod(value.shape)) for value in model.trainable_variables
    )
    writer = WeightVideoWriter(
        args.output,
        parameter_count,
        frame_rate=args.frame_rate,
        magnitude_maximum=args.magnitude_maximum,
        colormap=args.colormap,
    )
    started = time.time()
    writer.add(model)
    optimizer_step = 0
    epoch = 1
    try:
        while optimizer_step < args.steps:
            for batch in training_data.batches(
                args.batch_size, shuffle=True, seed=args.seed + epoch
            ):
                trainer.train_batch(batch)
                optimizer_step += 1
                if optimizer_step % args.frame_interval == 0:
                    writer.add(model)
                if optimizer_step % 5_000 == 0:
                    print(
                        f"step={optimizer_step} frames={writer.frame_count} "
                        f"seconds={time.time() - started:.1f}",
                        flush=True,
                    )
                if optimizer_step >= args.steps:
                    break
            epoch += 1
    finally:
        writer.close()

    reference_difference = None
    if args.reference_checkpoint is not None:
        reference = NeuralProgrammerInterpreter(SPEC)
        reference.build_for_task()
        reference.load_weights(args.reference_checkpoint)
        reference_difference = max(
            float(np.max(np.abs(actual.numpy() - expected.numpy())))
            for actual, expected in zip(
                model.trainable_variables, reference.trainable_variables, strict=True
            )
        )

    metadata = {
        "framework": "tensorflow",
        "tensorflow": tf.__version__,
        "task": SPEC.name,
        "training_distribution": "weighted connected Erdos-Renyi",
        "maximum_train_nodes": args.maximum_train_nodes,
        "training_examples_per_size": args.training_examples_per_size,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer_steps": args.steps,
        "seed": args.seed,
        "xla": not args.no_xla,
        "parameter_count": parameter_count,
        "raster": {
            "width": writer.width,
            "height": writer.height,
            "order": "trainable variables concatenated and flattened in row-major order",
            "variables": parameter_metadata(model, writer.width),
        },
        "video": {
            "file": args.output.name,
            "codec": "H.264 High 4:4:4 Predictive",
            "pixel_format": "yuv444p",
            "frame_rate": args.frame_rate,
            "frame_count": writer.frame_count,
            "frame_interval_steps": args.frame_interval,
            "step_for_frame": "frame_index * frame_interval_steps",
            "duration_seconds": writer.frame_count / args.frame_rate,
        },
        "color": {
            "value": "absolute parameter magnitude",
            "colormap": args.colormap,
            "normalization": "log1p(abs(weight) / 1e-4) / log1p(maximum / 1e-4)",
            "maximum": args.magnitude_maximum,
            "observed_maximum": writer.observed_maximum,
            "clipped_values_across_all_frames": writer.clipped_values,
        },
        "training_seconds": time.time() - started,
        "reference_checkpoint": (
            str(args.reference_checkpoint) if args.reference_checkpoint else None
        ),
        "maximum_reference_weight_difference": reference_difference,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"video={args.output}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
