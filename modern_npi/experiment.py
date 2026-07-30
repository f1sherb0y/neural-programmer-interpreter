import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from modern_npi.data import EpisodeBatches, select_episodes
from modern_npi.environment import decimal_add
from modern_npi.executor import ExecutionFailure, execute_addition
from modern_npi.model import NeuralProgrammerInterpreter
from modern_npi.traces import random_decimal, training_traces


DEFAULT_CHECKPOINT = Path("artifacts/addition.pt")
DEFAULT_RESULTS = Path("artifacts/addition_results.json")


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def to_tensors(batch: tuple[np.ndarray, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    tensors = []
    for index, value in enumerate(batch):
        tensor = torch.from_numpy(value)
        if index in (0, 7):
            tensor = tensor.float()
        else:
            tensor = tensor.long()
        tensors.append(tensor.to(device, non_blocking=True))
    return tuple(tensors)


def masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    ).reshape_as(mask)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


def decision_accuracy(
    end_logits: torch.Tensor,
    program_logits: torch.Tensor,
    argument_logits: tuple[torch.Tensor, ...],
    target_end: torch.Tensor,
    target_program: torch.Tensor,
    target_arguments: tuple[torch.Tensor, ...],
    child_mask: torch.Tensor,
) -> tuple[int, int]:
    correct = end_logits.argmax(dim=-1).eq(target_end)
    child_correct = program_logits.argmax(dim=-1).eq(target_program)
    for logits, target in zip(argument_logits, target_arguments, strict=True):
        child_correct.logical_and_(logits.argmax(dim=-1).eq(target))
    correct.logical_and_(child_mask.eq(0).logical_or(child_correct))
    return int(correct.sum().item()), correct.numel()


class Trainer:
    def __init__(
        self,
        model: NeuralProgrammerInterpreter,
        learning_rate: float,
        device: torch.device,
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_batch(self, batch: tuple[np.ndarray, ...]) -> tuple[float, int, int]:
        (
            features,
            programs,
            target_end,
            target_program,
            arg0,
            arg1,
            arg2,
            child_mask,
        ) = to_tensors(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        end_logits, program_logits, argument_logits = self.model(features, programs)
        end_loss = F.cross_entropy(end_logits.flatten(0, 1), target_end.flatten())
        program_loss = masked_cross_entropy(program_logits, target_program, child_mask)
        argument_loss = sum(
            masked_cross_entropy(logits, target, child_mask)
            for logits, target in zip(
                argument_logits,
                (arg0, arg1, arg2),
                strict=True,
            )
        )
        loss = end_loss + program_loss + argument_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        correct, count = decision_accuracy(
            end_logits,
            program_logits,
            argument_logits,
            target_end,
            target_program,
            (arg0, arg1, arg2),
            child_mask,
        )
        return float(loss.detach().item()), correct, count


@torch.inference_mode()
def teacher_forced_accuracy(
    model: NeuralProgrammerInterpreter,
    data: EpisodeBatches,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    count = 0
    for batch in data.batches(batch_size, shuffle=False, seed=0):
        (
            features,
            programs,
            target_end,
            target_program,
            arg0,
            arg1,
            arg2,
            child_mask,
        ) = to_tensors(batch, device)
        end_logits, program_logits, argument_logits = model(features, programs)
        batch_correct, batch_count = decision_accuracy(
            end_logits,
            program_logits,
            argument_logits,
            target_end,
            target_program,
            (arg0, arg1, arg2),
            child_mask,
        )
        correct += batch_correct
        count += batch_count
    return correct / count


def make_data(examples_per_length: int, maximum_length: int, seed: int) -> EpisodeBatches:
    traces = training_traces(examples_per_length, maximum_length, seed)
    episodes = [episode for trace in traces for episode in trace.episodes]
    return EpisodeBatches(select_episodes(episodes, seed=seed))


def save_model(
    model: NeuralProgrammerInterpreter,
    checkpoint: Path,
    args: argparse.Namespace,
) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "seed": args.seed,
            "maximum_train_length": args.maximum_train_length,
            "examples_per_length": args.examples_per_length,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        checkpoint,
    )


def train(
    args: argparse.Namespace,
    device: torch.device,
) -> NeuralProgrammerInterpreter:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_data = make_data(args.examples_per_length, args.maximum_train_length, args.seed)
    validation_examples = getattr(args, "validation_examples_per_length", 8)
    validation_data = make_data(
        validation_examples, args.maximum_train_length, args.seed + 1
    )
    model = NeuralProgrammerInterpreter().to(device)
    trainer = Trainer(model, args.learning_rate, device)

    print(
        f"device: {device}"
        + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
    )
    print(
        f"training: {train_data.size} invocations from "
        f"{args.examples_per_length} examples/length 1..{args.maximum_train_length}"
    )
    best_validation = -1.0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        batches = 0
        for batch in train_data.batches(
            args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
        ):
            loss, batch_correct, batch_count = trainer.train_batch(batch)
            loss_sum += loss
            correct += batch_correct
            count += batch_count
            batches += 1

        train_accuracy = correct / count
        validation_accuracy = teacher_forced_accuracy(
            model, validation_data, args.batch_size, device
        )
        print(
            f"epoch {epoch:02d}: loss={loss_sum / batches:.4f} "
            f"train_decision={train_accuracy:.6f} "
            f"validation_decision={validation_accuracy:.6f}"
        )
        if validation_accuracy >= best_validation:
            best_validation = validation_accuracy
            save_model(model, args.checkpoint, args)
        if train_accuracy == 1.0 and validation_accuracy == 1.0:
            break

    model = load_model(args.checkpoint, device)
    print(f"training_seconds: {time.time() - started:.1f}")
    return model


def load_model(
    checkpoint: Path,
    device: torch.device,
) -> NeuralProgrammerInterpreter:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model = NeuralProgrammerInterpreter().to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


@dataclass(frozen=True)
class LengthResult:
    length: int
    examples: int
    correct: int
    accuracy: float
    average_model_steps: float
    seconds: float
    first_failure: str | None


def evaluate_lengths(
    model: NeuralProgrammerInterpreter,
    lengths: list[int],
    examples: int,
    seed: int,
    device: torch.device,
) -> list[LengthResult]:
    rng = random.Random(seed)
    results = []
    for length in lengths:
        correct = 0
        steps = 0
        first_failure = None
        started = time.time()
        length_examples = examples if length <= 100 else min(examples, 2)
        for _ in range(length_examples):
            first = random_decimal(length, rng)
            second = random_decimal(length, rng)
            expected = decimal_add(first, second)
            try:
                execution = execute_addition(model, first, second, device)
                steps += execution.model_steps
                if execution.value == expected:
                    correct += 1
                elif first_failure is None:
                    first_failure = (
                        f"wrong result: expected {expected[:80]}, got {execution.value[:80]}"
                    )
            except ExecutionFailure as error:
                if first_failure is None:
                    first_failure = str(error)
        elapsed = time.time() - started
        result = LengthResult(
            length,
            length_examples,
            correct,
            correct / length_examples,
            steps / length_examples,
            elapsed,
            first_failure,
        )
        results.append(result)
        print(
            f"length {length:4d}: {correct}/{length_examples} "
            f"accuracy={result.accuracy:.3f} steps={result.average_model_steps:.1f} "
            f"seconds={elapsed:.1f}"
        )
        if first_failure:
            print(f"  first failure: {first_failure}")
    return results


def parse_lengths(value: str) -> list[int]:
    lengths = [int(item) for item in value.split(",")]
    if not lengths or any(length < 1 for length in lengths):
        raise argparse.ArgumentTypeError("Lengths must be positive integers")
    return lengths


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("train", "evaluate", "reproduce"))
    result.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    result.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    result.add_argument("--examples-per-length", type=int, default=32)
    result.add_argument("--maximum-train-length", type=int, default=20)
    result.add_argument("--epochs", type=int, default=60)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--device", default="auto")
    result.add_argument(
        "--lengths",
        type=parse_lengths,
        default=parse_lengths("1,5,10,20,50,100,500,1000,3000"),
    )
    result.add_argument("--evaluation-examples", type=int, default=16)
    return result


def main() -> None:
    args = parser().parse_args()
    device = choose_device(args.device)
    if args.command in ("train", "reproduce"):
        model = train(args, device)
    else:
        model = load_model(args.checkpoint, device)

    if args.command in ("evaluate", "reproduce"):
        results = evaluate_lengths(
            model,
            args.lengths,
            args.evaluation_examples,
            args.seed + 2,
            device,
        )
        payload = {
            "torch": torch.__version__,
            "device": str(device),
            "seed": args.seed,
            "training_examples_per_length": args.examples_per_length,
            "maximum_training_length": args.maximum_train_length,
            "boundary_feature": True,
            "results": [asdict(result) for result in results],
        }
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"results: {args.results}")


if __name__ == "__main__":
    main()
