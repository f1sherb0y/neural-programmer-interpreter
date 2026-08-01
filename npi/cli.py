import argparse
import sys

TASKS = ("addition", "graph", "graph-sweep")


def root_parser():
    parser = argparse.ArgumentParser(
        description="TensorFlow/XLA Neural Programmer-Interpreter"
    )
    parser.add_argument("task", choices=TASKS)
    return parser


def main():
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        root_parser().print_help()
        return
    task = sys.argv[1]
    if task not in TASKS:
        root_parser().error(f"unknown task: {task}")
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    if task == "addition":
        from npi.tasks.addition.experiment import main as task_main
    elif task == "graph":
        from npi.tasks.graph.experiment import main as task_main
    else:
        from npi.tasks.graph.sweep import main as task_main
    task_main()


if __name__ == "__main__":
    main()
