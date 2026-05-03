import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
from PIL import Image

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def png_bytes(color=(255, 0, 0), size=(8, 8)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


REQUIRED_COLUMNS = [
    "id",
    "year",
    "group",
    "points",
    "problem_number",
    "problem_statement",
    "answer",
    "multimodal",
    "sol_A",
    "sol_B",
    "sol_C",
    "sol_D",
    "sol_E",
    "question_image",
    "sol_A_image_bin",
    "sol_B_image_bin",
    "sol_C_image_bin",
    "sol_D_image_bin",
    "sol_E_image_bin",
    "associated_images_bin",
    "language",
]


def make_row(idx=1, with_images=False, invalid_question=False):
    qimg = None
    assoc = None
    if with_images:
        qimg = b"not-an-image" if invalid_question else png_bytes()
        assoc = [png_bytes((0, 255, 0))]
    row = {
        "id": idx,
        "year": 2024,
        "group": "Junior",
        "points": 5,
        "problem_number": idx,
        "problem_statement": f"Was ist 2+{idx}?",
        "answer": "A",
        "multimodal": bool(with_images),
        "sol_A": "3",
        "sol_B": "4",
        "sol_C": "5",
        "sol_D": "6",
        "sol_E": "7",
        "question_image": qimg,
        "sol_A_image_bin": None,
        "sol_B_image_bin": None,
        "sol_C_image_bin": None,
        "sol_D_image_bin": None,
        "sol_E_image_bin": None,
        "associated_images_bin": assoc,
        "language": "de",
    }
    return row


def write_parquet(tmp_path: Path, rows):
    # Ensure all required columns exist
    df = pd.DataFrame(rows)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[REQUIRED_COLUMNS]
    path = tmp_path / "data.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


class FakeResp:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        if text is None and payload is not None:
            text = json.dumps(payload)
        self.text = text or ""
        # Optional headers mapping for fallback parsing
        self.headers = {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def run_eval(
    monkeypatch,
    tmp_path: Path,
    dataset_path: Path,
    model_id: str,
    responses,
    extra_args: Optional[List[str]] = None,
    capture_payloads: Optional[List[Dict[str, Any]]] = None,
):
    # Import module fresh
    import importlib.util

    monkeypatch.syspath_prepend(str(ROOT))
    mod_path = ROOT / "eval_run.py"
    spec = importlib.util.spec_from_file_location("eval_run", mod_path)
    assert spec and spec.loader, "Could not load eval_run module spec"
    eval_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_run)  # type: ignore

    # Env
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    # Patch models registry to allow custom models if needed
    def fake_registry(_path):
        if model_id == "text-only/test":
            return {
                model_id: eval_run.ModelInfo(
                    id=model_id,
                    label="TextOnly",
                    supports_vision=False,
                    supports_json_response_format=True,
                    min_request_interval=None,
                )
            }
        return {
            model_id: eval_run.ModelInfo(
                id=model_id,
                label="GPT5",
                supports_vision=True,
                supports_json_response_format=True,
                min_request_interval=None,
            )
        }

    monkeypatch.setattr(eval_run, "read_models_registry", fake_registry)

    # Patch httpx.AsyncClient.post with queued responses
    calls = {"count": 0}

    async def fake_post(self, url, *, headers=None, json=None, timeout=None, **kwargs):
        i = calls["count"]
        calls["count"] += 1
        if capture_payloads is not None:
            capture_payloads.append(json)
        resp = responses[i] if i < len(responses) else responses[-1]
        return resp

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    # Prepare args
    out_dir = tmp_path / "runs"
    argv = [
        "eval_run.py",
        "--dataset",
        str(dataset_path),
        "--model",
        model_id,
        "--output_dir",
        str(out_dir),
        "--max_tokens",
        "64",
    ]
    if extra_args:
        argv.extend(extra_args)
    monkeypatch.setattr(sys, "argv", argv)

    # Run
    eval_run.main()

    # Find run dir
    safe_model = model_id.replace("/", "-")
    run_dirs = sorted(out_dir.glob(f"*_{safe_model}"))
    assert run_dirs, "run directory not created"
    run_dir = run_dirs[-1]
    return run_dir


def test_answer_json_success(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(1, with_images=False)])

    payload = {
        "id": "gen_1",
        "model": "openai/gpt-5",
        "choices": [{"message": {"content": '{"answer":"A"}'}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0.002,
        },
    }
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(200, payload)],
    )

    # Validate results
    import pandas as pd

    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["predicted"] == "A"
    assert bool(row["is_correct"]) is True
    assert row["reasoning_mode"] == "self-directed"

    config = json.loads((run_dir / "config.json").read_text())
    assert config["args"].get("reasoning") == "self-directed"


def test_no_live_dashboard_flag(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(1, with_images=False)])

    payload = {
        "id": "gen_flag",
        "model": "openai/gpt-5",
        "choices": [{"message": {"content": "A"}}],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
            "cost": 0.0015,
        },
    }

    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(200, payload)],
        extra_args=["--no-live-dashboard"],
    )

    events_file = run_dir / "usage_events.jsonl"
    assert events_file.exists(), (
        "usage_events.jsonl should be created when events logging is enabled"
    )
    with events_file.open("r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    assert lines, "usage events log should contain at least one entry"
    import pandas as pd

    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    row = df.iloc[0]
    assert row["points_earned"] == 5.0
    assert row["predicted"] == "A"
    assert row["total_tokens"] == 15
    assert abs(row["cost_usd"] - 0.0015) < 1e-9

    json_records = json.loads((run_dir / "results.json").read_text())
    assert len(json_records) == 1
    assert json_records[0]["predicted"] == "A"

    jsonl_lines = (run_dir / "results.jsonl").read_text().strip().splitlines()
    assert len(jsonl_lines) == 1
    assert json.loads(jsonl_lines[0])["predicted"] == "A"

    raw_responses = (run_dir / "raw_responses.jsonl").read_text().strip().splitlines()
    assert len(raw_responses) == 1
    payload = json.loads(raw_responses[0])
    assert payload["id"] == 1
    assert "choices" in payload["response"]
    assert payload["response"]["choices"][0]["message"]["content"]

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["answered_count"] == 1
    assert metrics["failed_count"] == 0
    assert metrics["accuracy"] == 1.0
    assert metrics["unknown_usage_count"] == 0
    assert metrics["declined_count"] == 0
    assert metrics["warning_row_count"] == 1
    assert metrics["warning_counts"].get("regex_fallback") == 1
    assert metrics["total_reasoning_tokens"] is None
    assert metrics["mean_reasoning_tokens"] is None
    assert metrics["reasoning_tokens_known_count"] == 0


def test_unparsed_response_penalized(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(3, with_images=False)])

    payload = {
        "id": "gen_blank",
        "model": "openai/gpt-5",
        "choices": [{"message": {"content": ""}}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
    }

    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(200, payload)],
    )

    import pandas as pd

    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    row = df.iloc[0]
    assert bool(row["is_correct"]) is False
    assert row["points_earned"] == pytest.approx(-1.25)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["total_points_earned"] == pytest.approx(-1.25)


def test_final_answer_line_precedence(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(21)])
    payload = {
        "id": "gen_final_line",
        "model": "openai/gpt-5",
        "choices": [
            {"message": {"content": "Here is my reasoning...\nFinal answer: C"}}
        ],
    }
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(200, payload)],
    )
    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    row = df.iloc[0]
    assert row["predicted"] == "C"
    assert row["warnings"] in (None, [])
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["declined_count"] == 0


def test_final_line_declined(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(22)])
    payload = {
        "id": "gen_declined_line",
        "model": "openai/gpt-5",
        "choices": [
            {
                "message": {
                    "content": "Unsure about this problem.\nFinal answer: Declined."
                }
            }
        ],
    }
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(200, payload)],
    )
    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    row = df.iloc[0]
    assert row["predicted"] == "DECLINED"
    assert row["points_earned"] == 0
    assert "declined_explicit" in (row["warnings"] or [])
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["declined_count"] == 1
    assert metrics["total_points_earned"] == pytest.approx(0.0)


def test_declined_phrase_detection(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(23)])
    payload = {
        "id": "gen_declined_phrase",
        "model": "openai/gpt-5",
        "choices": [
            {
                "message": {
                    "content": "After reviewing the options, I choose not to answer."
                }
            }
        ],
    }
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(200, payload)],
    )
    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    row = df.iloc[0]
    assert row["predicted"] == "DECLINED"
    assert "declined_phrase" in (row["warnings"] or [])


def test_declined_phrase_german(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(24)])
    payload = {
        "id": "gen_declined_de",
        "model": "openai/gpt-5",
        "choices": [
            {
                "message": {
                    "content": "Diese Aufgabe ist mehrdeutig; ich entscheide mich, nicht zu antworten."
                }
            }
        ],
    }
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(200, payload)],
    )
    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    row = df.iloc[0]
    assert row["predicted"] == "DECLINED"


def test_request_payload_has_no_response_format(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(25)])
    payload = {
        "id": "gen_payload",
        "model": "openai/gpt-5",
        "choices": [{"message": {"content": '{"answer":"A"}'}}],
    }
    captured: List[Dict[str, Any]] = []
    run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(200, payload)],
        capture_payloads=captured,
    )
    assert captured, "expected captured payload"
    first_payload = captured[0]
    assert "response_format" not in first_payload
    assert "reasoning" not in first_payload


def test_regex_fallback_and_missing_usage(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(3)])
    payload = {
        "id": "gen_3",
        "model": "openai/gpt-5",
        "choices": [{"message": {"content": "I think the answer is d."}}],
        # usage omitted
    }
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(200, payload)],
    )
    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    row = df.iloc[0]
    assert row["predicted"] == "D"
    assert pd.isna(row["total_tokens"]) or row["total_tokens"] is None

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["unknown_usage_count"] == 1
    assert metrics["warning_row_count"] == 1
    assert metrics["warning_counts"].get("regex_fallback") == 1
    assert metrics["total_reasoning_tokens"] is None
    assert metrics["mean_reasoning_tokens"] is None
    assert metrics["reasoning_tokens_known_count"] == 0


def test_list_content_response(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(7)])
    payload = {
        "id": "gen_list",
        "model": "openai/gpt-5",
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "output_text", "text": "Here is reasoning."},
                        {"type": "json", "json": {"answer": "A"}},
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
    }
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(200, payload)],
    )
    import pandas as pd

    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    row = df.iloc[0]
    assert row["predicted"] == "A"
    assert "json_extracted" in (row["warnings"] or [])

    raw_payload = json.loads((run_dir / "raw_responses.jsonl").read_text().strip())
    assert raw_payload["id"] == 7

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["warning_row_count"] == 1
    assert metrics["warning_counts"].get("json_extracted") == 1


def test_usage_details_in_response_json(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(8, with_images=False)])
    payload = {
        "id": "gen_json_usage",
        "model": "openai/gpt-5",
        "choices": [{"message": {"content": '{"answer":"A"}'}}],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
            "completion_tokens_details": {"reasoning_tokens": 42},
            "prompt_tokens_details": {"cached_tokens": 100, "audio_tokens": 0},
        },
    }
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(200, payload)],
    )
    import pandas as pd

    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    row = df.iloc[0]
    assert row["predicted"] == "A"
    assert row["total_tokens"] == 30
    assert row["reasoning_tokens"] == 42
    assert row["cached_prompt_tokens"] == 100
    assert row["audio_prompt_tokens"] == 0

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["total_reasoning_tokens"] == 42
    assert metrics["mean_reasoning_tokens"] == 42
    assert metrics["reasoning_tokens_known_count"] == 1


def test_x_usage_header_fallback(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(9, with_images=False)])
    payload = {
        "id": "gen_hdr_usage",
        "model": "openai/gpt-5",
        "choices": [{"message": {"content": '{"answer":"B"}'}}],
        # usage intentionally omitted to force header fallback
    }
    resp = FakeResp(200, payload)
    # Provide JSON-encoded usage in headers
    header_usage = {
        "prompt_tokens": 11,
        "completion_tokens": 6,
        "total_tokens": 17,
        "completion_tokens_details": {"reasoning_tokens": 5},
        "prompt_tokens_details": {"cached_tokens": 3, "audio_tokens": 1},
        "cost": 0.0015,
    }
    resp.headers = {"X-Usage": json.dumps(header_usage)}

    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[resp],
    )
    import pandas as pd

    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    row = df.iloc[0]
    assert row["predicted"] == "B"
    assert row["total_tokens"] == 17
    assert row["reasoning_tokens"] == 5
    assert row["cached_prompt_tokens"] == 3
    assert row["audio_prompt_tokens"] == 1

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["total_reasoning_tokens"] == 5
    assert metrics["mean_reasoning_tokens"] == 5
    assert metrics["reasoning_tokens_known_count"] == 1

    # Now verify non-JSON key=value format also parses
    payload2 = {
        "id": "gen_hdr_usage2",
        "model": "openai/gpt-5",
        "choices": [{"message": {"content": '{"answer":"C"}'}}],
    }
    resp2 = FakeResp(200, payload2)
    kv = (
        "prompt_tokens=7, completion_tokens=5, total_tokens=12, cost=0.001, "
        "completion_tokens_details.reasoning_tokens=2, prompt_tokens_details.cached_tokens=4, prompt_tokens_details.audio_tokens=0"
    )
    resp2.headers = {"X-Usage": kv}
    run_dir2 = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[resp2],
    )
    df2 = pd.read_parquet(run_dir2 / "results.parquet", engine="pyarrow")
    row2 = df2.iloc[0]
    assert row2["predicted"] == "C"
    assert row2["total_tokens"] == 12
    assert row2["reasoning_tokens"] == 2
    assert row2["cached_prompt_tokens"] == 4
    assert row2["audio_prompt_tokens"] == 0

    metrics2 = json.loads((run_dir2 / "metrics.json").read_text())
    assert metrics2["total_reasoning_tokens"] == 2
    assert metrics2["mean_reasoning_tokens"] == 2
    assert metrics2["reasoning_tokens_known_count"] == 1


def test_http_error_failure(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(4)])
    # Use HTTP 400 which is not retryable, avoiding the retry delays
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(400, payload=None, text="Bad request")],
    )
    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    row = df.iloc[0]
    assert row["error"].startswith("http_error_400")
    assert pd.isna(row["predicted"]) or row["predicted"] is None

    failures = (run_dir / "failures.jsonl").read_text().strip().splitlines()
    assert failures and json.loads(failures[0])["error"].startswith("http_error_400")


def test_skip_images_for_text_only_model(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(5, with_images=True)])
    # Use custom text-only model
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="text-only/test",
        responses=[],
    )
    # No results because the only row was filtered out pre-run
    import pandas as pd

    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    assert len(df) == 0
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["answered_count"] == 0
    # Now filtered at load time, not skipped at worker time
    assert metrics["skipped_count"] == 0
    assert metrics["text_only_evaluation"] is True
    assert metrics["text_only_source"] == "model"
    assert metrics.get("model_filtered_out_multimodal_rows") == 1


def test_text_only_cli_filters_multimodal(monkeypatch, tmp_path):
    dataset = write_parquet(tmp_path, [make_row(10, with_images=True)])
    # Vision-capable model, but CLI flag should filter out multimodal rows
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[],
        extra_args=["--text-only"],
    )
    import pandas as pd

    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    assert len(df) == 0
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["answered_count"] == 0
    assert metrics["skipped_count"] == 0
    assert metrics["text_only_evaluation"] is True
    assert metrics["text_only_source"] == "cli"
    assert metrics.get("cli_filtered_out_multimodal_rows") == 1


def test_image_decode_warning(monkeypatch, tmp_path):
    dataset = write_parquet(
        tmp_path, [make_row(6, with_images=True, invalid_question=True)]
    )
    payload = {
        "id": "gen_6",
        "model": "openai/gpt-5",
        "choices": [{"message": {"content": '{"answer":"A"}'}}],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 4,
            "total_tokens": 12,
            "cost": 0.001,
        },
    }
    run_dir = run_eval(
        monkeypatch,
        tmp_path,
        dataset,
        model_id="openai/gpt-5",
        responses=[FakeResp(200, payload)],
    )
    df = pd.read_parquet(run_dir / "results.parquet", engine="pyarrow")
    row = df.iloc[0]
    warnings = row["warnings"]
    # Stored as list in parquet via pandas -> may deserialize as list
    assert warnings and "question_image_decode_failed" in warnings
