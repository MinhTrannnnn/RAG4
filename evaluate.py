import argparse
import sys
from types import SimpleNamespace

from scripts.teacher_client import command_evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Teacher Server evaluation")
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Send document_received=true after vector DB has already been built.",
    )
    args = parser.parse_args()
    try:
        command_evaluate(SimpleNamespace(skip_upload=args.skip_upload))
    except RuntimeError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
