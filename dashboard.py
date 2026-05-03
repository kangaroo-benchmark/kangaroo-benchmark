"""Dashboard entrypoint launching the FastAPI app."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from dashboard.server import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the benchmark dashboard server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument(
        "--runs-dir", default="runs", help="Directory containing run folders"
    )
    parser.add_argument(
        "--models", default="models.json", help="Path to models registry JSON"
    )
    parser.add_argument(
        "--templates", default="web/templates", help="Template directory"
    )
    parser.add_argument(
        "--static", default="web/static", help="Static assets directory"
    )
    parser.add_argument(
        "--human-results",
        default="human_results",
        help="Directory containing human baseline JSONs",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload (development only)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        runs_dir=Path(args.runs_dir),
        models_path=Path(args.models),
        templates_dir=Path(args.templates),
        static_dir=Path(args.static),
        human_results_dir=Path(args.human_results),
    )
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
