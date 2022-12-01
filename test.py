"""
    Module to run the model evaluations.
"""
import argparse


if __name__ == "__main__":
    MODEL_SIZES = ["tiny", "small", "large", "base"]
    arg_parser = argparse.ArgumentParser(
        description="Evaluates a chosen model for the provided data.",
        add_help=False,
        usage=f"test.py path [--model {{ {', '.join(MODEL_SIZES)} }}]",
    )
    arg_parser.add_argument("path")
    arg_parser.add_argument("--model", choices=MODEL_SIZES, default="tiny")

    args = arg_parser.parse_args()
    print(args)
