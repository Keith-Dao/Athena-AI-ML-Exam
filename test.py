"""
    Module to run the model evaluations.
"""
import argparse
from src.evaluator import Evaluator


class ArgumentParser(argparse.ArgumentParser):
    """
    Argument parser for this module.
    """

    def __init__(
        self, model_choices: list[str], default_model_choice: str
    ) -> None:
        super().__init__(
            description="Evaluates a chosen model for the provided data.",
            add_help=False,
            usage=f"test.py path [--model {{ {', '.join(model_choices)} }}]",
        )
        self.add_argument("path")
        self.add_argument(
            "--model", choices=model_choices, default=default_model_choice
        )


if __name__ == "__main__":
    parser = ArgumentParser(
        model_choices=Evaluator.model_choices,
        default_model_choice=Evaluator.default_model_choice,
    )
    args = parser.parse_args()

    Evaluator(args.path, args.model)
