"""
    Module to run the model evaluations.
"""
import argparse


class ArgumentParser(argparse.ArgumentParser):
    """
    Argument parser for this module.
    """

    def __init__(self, model_choices: list[str]) -> None:
        super().__init__(
            description="Evaluates a chosen model for the provided data.",
            add_help=False,
            usage=f"test.py path [--model {{ {', '.join(model_choices)} }}]",
        )
        self.add_argument("path")
        self.add_argument("--model", choices=model_choices, default="tiny")


if __name__ == "__main__":
    MODEL_SIZES = ["tiny", "small", "large", "base"]
    parser = ArgumentParser(model_choices=MODEL_SIZES)
    args = parser.parse_args()
