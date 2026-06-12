import argparse
from types import SimpleNamespace

from scripts.teacher_client import command_register


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register this Student Server")
    parser.add_argument("--server-url", default=None)
    args = parser.parse_args()
    command_register(SimpleNamespace(server_url=args.server_url))
