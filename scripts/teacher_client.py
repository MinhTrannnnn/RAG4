import argparse
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

TEACHER_BASE_URL = os.getenv("TEACHER_BASE_URL", "http://192.168.50.218:8000/api/v1")
STUDENT_ID = os.getenv("STUDENT_ID", "B22DCVT351")
STUDENT_PORT = int(os.getenv("STUDENT_PORT", "5000"))
MY_SERVER_URL = os.getenv("MY_SERVER_URL", "").strip().rstrip("/")
VECTOR_DB_PATH = Path(os.getenv("VECTOR_DB_PATH", "data/vector_db.pkl"))
if not VECTOR_DB_PATH.is_absolute():
    VECTOR_DB_PATH = (PROJECT_ROOT / VECTOR_DB_PATH).resolve()


def guess_lan_ip() -> str:
    teacher_host = urlparse(TEACHER_BASE_URL).hostname or ""
    teacher_match = re.match(r"^(\d+\.\d+\.\d+)\.\d+$", teacher_host)
    teacher_prefix = teacher_match.group(1) if teacher_match else None

    if teacher_prefix:
        try:
            output = subprocess.check_output(
                ["ipconfig"],
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            for match in re.finditer(r"IPv4 Address[^\r\n:]*:\s*(\d+\.\d+\.\d+\.\d+)", output):
                ip = match.group(1)
                if ip.startswith(f"{teacher_prefix}.") and not ip.endswith(".0"):
                    return ip
        except (OSError, subprocess.SubprocessError):
            pass

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("192.168.50.218", 8000))
            return sock.getsockname()[0]
        except OSError:
            return socket.gethostbyname(socket.gethostname())


def request_json(method: str, path: str, body: dict | None = None, timeout: int = 180):
    url = f"{TEACHER_BASE_URL}{path}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {
        "X-Student-ID": STUDENT_ID,
    }

    if body is not None:
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=method)

    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
            if not response_body:
                return {}
            return json.loads(response_body)
    except HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {message}") from error
    except URLError as error:
        raise RuntimeError(f"Cannot reach Teacher Server: {error.reason}") from error


def print_json(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def command_register(args):
    ensure_config()
    server_url = args.server_url or MY_SERVER_URL
    if not server_url:
        server_url = f"http://{guess_lan_ip()}:{STUDENT_PORT}"

    print(f"Registering Student Server: {server_url}")
    result = request_json(
        "POST",
        "/competition/register",
        {"server_url": server_url},
    )
    print_json(result)


def command_evaluate(args):
    ensure_config()
    document_received = bool(getattr(args, "skip_upload", False))
    if document_received and (
        not VECTOR_DB_PATH.is_file() or VECTOR_DB_PATH.stat().st_size == 0
    ):
        raise RuntimeError(
            f"Cannot use --skip-upload because vector DB is missing: {VECTOR_DB_PATH}. "
            "Run python evaluate.py once without --skip-upload first."
        )

    print(f"Starting evaluation with document_received={str(document_received).lower()}")
    result = request_json(
        "POST",
        "/competition/evaluate",
        {"document_received": document_received},
        # 100 questions x up to 60 seconds, plus the optional 2-minute upload.
        timeout=7200,
    )
    print_json(result)


def command_result(_args):
    ensure_config()
    result = request_json("GET", "/competition/result")
    print_json(result)


def command_reset(_args):
    ensure_config()
    result = request_json("POST", "/competition/reset")
    print_json(result)


def ensure_config():
    if STUDENT_ID == "CHANGE_ME":
        raise RuntimeError("Please set STUDENT_ID in .env first.")


def main():
    parser = argparse.ArgumentParser(description="Teacher Server CLI for the RAG exam")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--server-url", default=None)
    register_parser.set_defaults(func=command_register)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Send document_received=true after vector DB has already been built.",
    )
    evaluate_parser.set_defaults(func=command_evaluate)

    result_parser = subparsers.add_parser("result")
    result_parser.set_defaults(func=command_result)

    reset_parser = subparsers.add_parser("reset")
    reset_parser.set_defaults(func=command_reset)

    args = parser.parse_args()

    try:
        args.func(args)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
