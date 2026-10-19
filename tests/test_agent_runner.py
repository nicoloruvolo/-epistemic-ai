"""Offline tests for agent_runner.py. No network, API keys, or paid calls.

An autouse fixture blocks all socket connections, so any accidental real API
call fails the test instead of reaching the network.
"""

from __future__ import annotations

import importlib
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent_runner as ar  # noqa: E402

FAKE_OPENAI_KEY = "sk-test-SECRETKEY1234567890"
OPTIONS = ["Prescribe antibiotics now", "Request additional information", "Refer to specialist"]

# Sentinels placed in evaluator-only fields; they must never reach the target.
SENTINELS = [
    "SENTINEL_RATIONALE_9f2",
    "SENTINEL_EQUIV_NOTE_41c",
    "SENTINEL_REVIEWER_77a",
    "2031-01-01T00:00:00+00:00",
]


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def guard(*args, **kwargs):
        raise AssertionError("Network access attempted during an offline test.")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TARGET_PROVIDER", "TARGET_MODEL",
                 "TARGET_TEMPERATURE", "API_TIMEOUT_SECONDS", "API_MAX_RETRIES",
                 "TARGET_MAX_OUTPUT_TOKENS"):
        monkeypatch.delenv(name, raising=False)


def make_scenario(sid="S01", *, review="approved", variants=None, options=None, acceptable=None):
    options = options or list(OPTIONS)
    return {
        "scenario_id": sid,
        "domain": "medical",
        "difficulty": "low",
        "scenario_text": f"Scenario text for {sid}. A fictional patient presents with mild symptoms.",
        "decision_options": options,
        "acceptable_decisions": acceptable or [options[1]],
        "scoring_rationale": SENTINELS[0],
        "review_status": review,
        "reviewer": SENTINELS[2] if review == "approved" else None,
        "reviewed_at": SENTINELS[3] if review == "approved" else None,
        "variants": variants if variants is not None else [
            {"variant_id": f"{sid}-V1", "variant_type": "semantic_paraphrase",
             "scenario_text": f"Paraphrased text for {sid}.", "generator_equivalence_note": SENTINELS[1],
             "equivalence_review_status": "approved", "reviewer": "rev", "reviewed_at": "2031-01-01"},
            {"variant_id": f"{sid}-V2", "variant_type": "irrelevant_pressure",
             "scenario_text": f"Urgent pressure text for {sid}.", "generator_equivalence_note": SENTINELS[1],
             "equivalence_review_status": "pending_review", "reviewer": None, "reviewed_at": None},
        ],
    }


def make_dataset(*scenarios):
    return {"dataset_version": "0.1", "generated_at": "2026-01-01T00:00:00+00:00",
            "generator_model": "gen-model", "scientific_notice": "x",
            "scenarios": list(scenarios) or [make_scenario()]}


def write_dataset(tmp_path, dataset) -> Path:
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    return path


def make_config(**overrides):
    values = dict(target_provider="openai", target_model="target-x", api_key=FAKE_OPENAI_KEY,
                  timeout_seconds=30.0, max_retries=2, temperature=0.0, max_output_tokens=600)
    values.update(overrides)
    return ar.RunnerConfig(**values)


def good_output(decision=OPTIONS[1], confidence=0.8, rationale="because"):
    return ar.RawModelOutput(
        text=json.dumps({"decision": decision, "confidence_score": confidence, "rationale": rationale}),
        stop_reason="stop",
    )


class FakeClient:
    """Returns scripted outputs or raises scripted ProviderErrors, in order."""

    provider = "fake"

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item


def first_call(plan_scenario=None, demo=False):
    ds = ar.RunnerDataset.model_validate(make_dataset(plan_scenario or make_scenario()))
    return ar.build_run_plan(ds, demo).calls[0]


# ---------------------------------------------------------------- import / help
def test_import_makes_no_api_call_or_client():
    importlib.reload(ar)  # network is blocked; reload must not construct clients


def test_help_exits_without_calling_api(capsys):
    def exploding_factory(config):
        raise AssertionError("client_factory must not be called for --help")

    with pytest.raises(SystemExit) as info:
        ar.main(["--help"], client_factory=exploding_factory)
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "paid API calls" in out and "--force" in out


# ---------------------------------------------------------------- target payload
def test_target_payload_contains_no_evaluator_only_fields():
    call = first_call()
    request = ar.build_target_request(call, make_config())
    payload = json.dumps({
        "system": request.system_prompt, "user": request.user_prompt,
        "schema": request.response_schema(), "fingerprint": request.fingerprint(),
    })
    for sentinel in SENTINELS:
        assert sentinel not in payload
    for key_name in ("acceptable_decisions", "scoring_rationale", "review_status",
                     "reviewer", "generator_equivalence_note", "equivalence_review_status"):
        assert key_name not in payload
    # What the target should see:
    assert call.scenario_text in payload
    assert all(option in payload for option in OPTIONS)


def test_runner_models_do_not_retain_answer_key_fields(tmp_path):
    dataset, _ = ar.load_dataset(write_dataset(tmp_path, make_dataset()))
    dumped = dataset.model_dump_json()
    for forbidden in ("acceptable_decisions", "scoring_rationale", "generator_equivalence_note", SENTINELS[0], SENTINELS[1]):
        assert forbidden not in dumped


def test_sdk_call_kwargs_contain_no_evaluator_fields():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=good_output().text, refusal=None), finish_reason="stop")])

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    target = ar.OpenAITarget(make_config(), sdk_client=sdk)
    request = ar.build_target_request(first_call(), make_config())
    out = target.complete(request)
    blob = json.dumps(captured, default=str)
    assert not any(s in blob for s in SENTINELS)
    assert captured["response_format"]["json_schema"]["schema"]["properties"]["decision"]["enum"] == OPTIONS
    assert captured["max_completion_tokens"] == 600 and captured["temperature"] == 0.0
    assert out.text == good_output().text


def test_anthropic_adapter_extracts_forced_tool_input():
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        block = SimpleNamespace(type="tool_use", input={"decision": OPTIONS[0], "confidence_score": 0.6, "rationale": "r"})
        return SimpleNamespace(content=[block], stop_reason="tool_use")

    sdk = SimpleNamespace(messages=SimpleNamespace(create=create))
    cfg = make_config(target_provider="anthropic", temperature=None)
    out = ar.AnthropicTarget(cfg, sdk_client=sdk).complete(ar.build_target_request(first_call(), cfg))
    assert "temperature" not in seen and seen["tool_choice"]["name"] == "submit_decision"
    assert ar.parse_model_answer(out.text, OPTIONS).decision == OPTIONS[0]
    assert not any(s in json.dumps(seen, default=str) for s in SENTINELS)


# ---------------------------------------------------------------- dataset validation
def test_valid_dataset_loads(tmp_path):
    dataset, digest = ar.load_dataset(write_dataset(tmp_path, make_dataset(make_scenario("S01"), make_scenario("S02"))))
    assert len(dataset.scenarios) == 2 and len(digest) == 64


@pytest.mark.parametrize("mutate, fragment", [
    (lambda d: d["scenarios"][0].pop("scenario_text"), "scenario_text"),
    (lambda d: d["scenarios"][0].pop("review_status"), "review_status"),
    (lambda d: d["scenarios"][0].update(review_status="maybe"), "review_status"),
    (lambda d: d["scenarios"][0].update(decision_options=["only one"]), "at least 2"),
    (lambda d: d["scenarios"][0].update(decision_options=["a", "a", "b"]), "unique"),
    (lambda d: d["scenarios"][0].update(decision_options=["a", " ", "b"]), "blank"),
    (lambda d: d["scenarios"][0]["variants"][0].pop("variant_id"), "variant_id"),
    (lambda d: d["scenarios"][0]["variants"][0].update(scenario_text="  "), "blank"),
    (lambda d: d["scenarios"][0]["variants"][0].update(equivalence_review_status="bogus"), "equivalence_review_status"),
    (lambda d: d["scenarios"][0].update(acceptable_decisions=["not an option"]), "integrity"),
    (lambda d: d["scenarios"][0].pop("acceptable_decisions"), "integrity"),
    (lambda d: d.update(scenarios=[]), "scenarios"),
    (lambda d: d.pop("generator_model"), "generator_model"),
])
def test_malformed_datasets_rejected(tmp_path, mutate, fragment):
    data = make_dataset()
    mutate(data)
    with pytest.raises(ar.DatasetError) as info:
        ar.load_dataset(write_dataset(tmp_path, data))
    assert fragment in str(info.value)


def test_duplicate_scenario_ids_rejected(tmp_path):
    with pytest.raises(ar.DatasetError, match="Duplicate scenario_id"):
        ar.load_dataset(write_dataset(tmp_path, make_dataset(make_scenario("S01"), make_scenario("S01"))))


def test_duplicate_variant_ids_rejected(tmp_path):
    data = make_dataset()
    data["scenarios"][0]["variants"][1]["variant_id"] = data["scenarios"][0]["variants"][0]["variant_id"]
    with pytest.raises(ar.DatasetError, match="Duplicate variant_id"):
        ar.load_dataset(write_dataset(tmp_path, data))


def test_variant_id_colliding_with_base_id_rejected(tmp_path):
    data = make_dataset()
    data["scenarios"][0]["variants"][0]["variant_id"] = "S01-BASE"
    with pytest.raises(ar.DatasetError, match="Duplicate variant_id"):
        ar.load_dataset(write_dataset(tmp_path, data))


def test_missing_and_invalid_json_files(tmp_path):
    with pytest.raises(ar.DatasetError, match="not found"):
        ar.load_dataset(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ar.DatasetError, match="not valid JSON"):
        ar.load_dataset(bad)


# ---------------------------------------------------------------- review gate
def test_default_plan_runs_only_approved_inputs():
    ds = ar.RunnerDataset.model_validate(make_dataset(
        make_scenario("S01", review="approved"),
        make_scenario("S02", review="pending_review"),
        make_scenario("S03", review="rejected"),
    ))
    plan = ar.build_run_plan(ds, allow_unreviewed_demo=False)
    assert [c.variant_id for c in plan.calls] == ["S01-BASE", "S01-V1"]  # V2 equivalence is pending
    assert not plan.provisional and not any(c.provisional for c in plan.calls)
    skipped = {s.variant_id for s in plan.skipped}
    assert {"S01-V2", "S02-BASE", "S02-V1", "S03-BASE"} <= skipped


def test_default_run_refuses_when_nothing_is_approved():
    ds = ar.RunnerDataset.model_validate(make_dataset(make_scenario("S01", review="pending_review")))
    with pytest.raises(ar.AgentRunnerError, match="allow-unreviewed-demo"):
        ar.build_run_plan(ds, allow_unreviewed_demo=False)


def test_approved_without_audit_trail_is_not_treated_as_reviewed():
    data = make_dataset(make_scenario("S01", review="approved"))
    data["scenarios"][0]["reviewer"] = None
    ds = ar.RunnerDataset.model_validate(data)
    with pytest.raises(ar.AgentRunnerError):
        ar.build_run_plan(ds, False)
    demo = ar.build_run_plan(ds, True)
    assert all(c.provisional for c in demo.calls)


def test_demo_mode_marks_unreviewed_inputs_provisional_and_never_runs_rejections():
    data = make_dataset(make_scenario("S01", review="pending_review"), make_scenario("S02", review="rejected"))
    data["scenarios"][0]["variants"][1]["equivalence_review_status"] = "not_equivalent"
    plan = ar.build_run_plan(ar.RunnerDataset.model_validate(data), True)
    assert plan.provisional
    assert [c.variant_id for c in plan.calls] == ["S01-BASE", "S01-V1"]
    assert all(c.provisional for c in plan.calls)
    assert {"S01-V2", "S02-BASE"} <= {s.variant_id for s in plan.skipped}


# ---------------------------------------------------------------- response validation
@pytest.mark.parametrize("confidence", [-0.1, 1.0001, 5, "0.9", True, None])
def test_invalid_confidence_rejected(confidence):
    raw = json.dumps({"decision": OPTIONS[0], "confidence_score": confidence, "rationale": "r"})
    with pytest.raises(ar.InvalidModelResponse) as info:
        ar.parse_model_answer(raw, OPTIONS)
    assert info.value.error_type == "invalid_confidence"


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_constants_rejected(literal):
    # Non-standard JSON constants are refused at the JSON layer, before schema validation.
    with pytest.raises(ar.InvalidModelResponse) as info:
        ar.parse_model_answer('{"decision": "%s", "confidence_score": %s}' % (OPTIONS[0], literal), OPTIONS)
    assert info.value.error_type == "invalid_json"


def test_missing_confidence_rejected():
    with pytest.raises(ar.InvalidModelResponse) as info:
        ar.parse_model_answer(json.dumps({"decision": OPTIONS[0]}), OPTIONS)
    assert info.value.error_type == "invalid_confidence"


@pytest.mark.parametrize("confidence", [0, 0.0, 0.5, 1, 1.0])
def test_boundary_confidences_accepted(confidence):
    raw = json.dumps({"decision": OPTIONS[0], "confidence_score": confidence})
    assert ar.parse_model_answer(raw, OPTIONS).confidence_score == confidence


@pytest.mark.parametrize("decision", ["Not an option", OPTIONS[0].lower(), OPTIONS[0] + " ", "1. " + OPTIONS[0], "", None])
def test_invalid_decisions_rejected_without_normalisation(decision):
    raw = json.dumps({"decision": decision, "confidence_score": 0.5})
    with pytest.raises(ar.InvalidModelResponse) as info:
        ar.parse_model_answer(raw, OPTIONS)
    assert info.value.error_type in ("invalid_decision", "schema_violation")


@pytest.mark.parametrize("text, kind", [
    ("", "empty_response"), (None, "empty_response"),
    ("not json", "invalid_json"),
    ('```json\n{"decision": "x", "confidence_score": 0.5}\n```', "invalid_json"),
    ("[1, 2]", "schema_violation"),
    (json.dumps({"decision": OPTIONS[0], "confidence_score": 0.5, "extra": 1}), "schema_violation"),
])
def test_malformed_responses_rejected(text, kind):
    with pytest.raises(ar.InvalidModelResponse) as info:
        ar.parse_model_answer(text, OPTIONS)
    assert info.value.error_type == kind


# ---------------------------------------------------------------- evaluate_call
POLICY = ar.RetryPolicy(max_retries=2)


def run_one(client, **cfg):
    return ar.evaluate_call(first_call(), client, make_config(**cfg), "run-test", POLICY, sleep=lambda s: None, jitter=lambda d: 0.0)


def test_abstention_is_a_successful_observation_not_an_api_failure():
    abstain = run_one(FakeClient([good_output(decision="Request additional information", confidence=0.9)]))
    assert abstain.status == "success"
    assert abstain.decision == "Request additional information"
    assert abstain.error_type is None and abstain.error_message is None

    failed = run_one(FakeClient([ar.ProviderError("bad_request", "rejected")]))
    assert failed.status == "api_error"
    assert failed.decision is None and failed.confidence_score is None
    assert failed.error_type == "bad_request"


def test_api_failure_records_no_fabricated_observation():
    record = run_one(FakeClient([ar.ProviderError("server_error", "boom", status_code=500)]))
    assert (record.status, record.decision, record.confidence_score, record.rationale) == ("api_error", None, None, None)
    assert record.attempts == 3  # 1 try + 2 retries, then recorded


def test_invalid_model_output_marked_invalid_response_with_raw_kept():
    record = run_one(FakeClient([good_output(confidence=1.7)]))
    assert record.status == "invalid_response" and record.error_type == "invalid_confidence"
    assert record.decision is None and record.confidence_score is None
    assert "1.7" in record.raw_response  # raw preserved for audit, not repaired


@pytest.mark.parametrize("raw, kind", [
    (ar.RawModelOutput(text=None, stop_reason="stop", refusal="I can't help"), "model_refusal"),
    (ar.RawModelOutput(text=good_output().text, stop_reason="length"), "truncated_output"),
    (ar.RawModelOutput(text=good_output().text, stop_reason="max_tokens"), "truncated_output"),
    (ar.RawModelOutput(text=None, stop_reason="content_filter"), "content_filtered"),
    (ar.RawModelOutput(text=None, stop_reason="stop"), "empty_response"),
])
def test_refusal_truncation_filter_and_empty_are_explicit_errors(raw, kind):
    record = run_one(FakeClient([raw]))
    assert record.status == "invalid_response" and record.error_type == kind and record.decision is None


def test_record_model_rejects_inconsistent_states():
    base = dict(run_id="r", scenario_id="S01", variant_id="S01-BASE", variant_type="base", provider="p", model="m",
                timestamp="t", attempts=1, provisional=False, scenario_review_status="approved",
                prompt_version="v", request_sha256="h")
    with pytest.raises(ValueError):
        ar.RunLogRecord(**base, status="success")
    with pytest.raises(ValueError):
        ar.RunLogRecord(**base, status="api_error", error_type="timeout", decision="x", confidence_score=0.5)
    with pytest.raises(ValueError):
        ar.RunLogRecord(**base, status="api_error")


# ---------------------------------------------------------------- retries
def test_retries_are_bounded_and_back_off():
    sleeps = []
    client = FakeClient([ar.ProviderError("rate_limit", "slow down")])
    policy = ar.RetryPolicy(max_retries=3)
    with pytest.raises(ar.ProviderError) as info:
        ar.call_with_retries(client, ar.build_target_request(first_call(), make_config()), policy,
                             sleep=sleeps.append, jitter=lambda d: 0.0)
    assert len(client.requests) == 4  # 1 initial + exactly 3 retries
    assert info.value.attempts == 4
    assert sleeps == [1.0, 2.0, 4.0]
    assert all(s <= policy.max_delay * 1.25 for s in sleeps)


def test_retry_after_lengthens_delay_but_is_capped():
    policy = ar.RetryPolicy(max_retries=1)
    assert ar.compute_delay(policy, 0, 7.0, lambda d: 0.0) == 7.0
    assert ar.compute_delay(policy, 0, 10_000.0, lambda d: 0.0) == policy.retry_after_cap
    assert ar.compute_delay(policy, 0, None, lambda d: 0.0) == 1.0
    assert ar.compute_delay(policy, 20, None, lambda d: 0.0) == policy.max_delay


def test_retry_succeeds_after_transient_error():
    client = FakeClient([ar.ProviderError("timeout", "t"), good_output()])
    raw, attempts = ar.call_with_retries(client, ar.build_target_request(first_call(), make_config()), POLICY,
                                         sleep=lambda s: None, jitter=lambda d: 0.0)
    assert attempts == 2 and raw.text


@pytest.mark.parametrize("kind", ["authentication", "permission", "model_not_found", "quota_exhausted", "bad_request", "unknown"])
def test_permanent_errors_are_not_retried(kind):
    client = FakeClient([ar.ProviderError(kind, "nope")])
    with pytest.raises(ar.ProviderError):
        ar.call_with_retries(client, ar.build_target_request(first_call(), make_config()), POLICY, sleep=lambda s: None)
    assert len(client.requests) == 1


def test_sdk_clients_are_built_with_sdk_retries_disabled():
    import anthropic
    import openai
    o = ar.OpenAITarget(make_config())
    assert o._client.max_retries == 0
    a = ar.AnthropicTarget(make_config(target_provider="anthropic", temperature=None))
    assert a._client.max_retries == 0
    assert isinstance(o._client, openai.OpenAI) and isinstance(a._client, anthropic.Anthropic)


def test_retry_after_header_parsing():
    assert ar.parse_retry_after({"Retry-After": "7"}) == 7.0
    assert ar.parse_retry_after({"retry-after-ms": "1500"}) == 1.5
    assert ar.parse_retry_after({"retry-after": "garbage"}) is None
    assert ar.parse_retry_after({}) is None and ar.parse_retry_after(None) is None


# ---------------------------------------------------------------- SDK error translation
def _sdk_exc(sdk, name, status, message="msg", headers=None, code=None):
    import httpx
    response = httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "http://test.invalid"))
    cls = getattr(sdk, name)
    exc = cls(message, response=response, body={"code": code} if code else None)
    return exc


@pytest.mark.parametrize("sdk_name", ["openai", "anthropic"])
def test_error_translation(sdk_name):
    sdk = importlib.import_module(sdk_name)
    rate = ar.translate_sdk_error(_sdk_exc(sdk, "RateLimitError", 429, headers={"retry-after": "9"}), sdk)
    assert (rate.kind, rate.retry_after_seconds, rate.retryable) == ("rate_limit", 9.0, True)
    assert ar.translate_sdk_error(_sdk_exc(sdk, "AuthenticationError", 401), sdk).fatal
    assert ar.translate_sdk_error(_sdk_exc(sdk, "NotFoundError", 404), sdk).kind == "model_not_found"
    assert ar.translate_sdk_error(_sdk_exc(sdk, "BadRequestError", 400), sdk).retryable is False
    assert ar.translate_sdk_error(_sdk_exc(sdk, "InternalServerError", 500), sdk).kind == "server_error"
    assert ar.translate_sdk_error(_sdk_exc(sdk, "BadRequestError", 400, message="Your credit balance is too low"), sdk).kind == "quota_exhausted"


def test_openai_insufficient_quota_is_fatal_not_retryable():
    import openai
    exc = _sdk_exc(openai, "RateLimitError", 429, code="insufficient_quota")
    translated = ar.translate_sdk_error(exc, openai)
    assert translated.kind == "quota_exhausted" and translated.fatal and not translated.retryable


# ---------------------------------------------------------------- secrets
def test_secrets_never_appear_in_errors_records_or_manifest(monkeypatch, tmp_path):
    import openai
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
    # Provider messages can echo keys; both fixed and redacted paths must be clean.
    auth = ar.translate_sdk_error(_sdk_exc(openai, "AuthenticationError", 401, message=f"Incorrect API key provided: {FAKE_OPENAI_KEY}"), openai)
    bad = ar.translate_sdk_error(_sdk_exc(openai, "BadRequestError", 400, message=f"oops {FAKE_OPENAI_KEY} and sk-proj-ABCDEFGH12345"), openai)
    for err in (auth, bad):
        assert FAKE_OPENAI_KEY not in err.message and "sk-proj-ABCDEFGH12345" not in err.message

    config = make_config()
    assert FAKE_OPENAI_KEY not in repr(config) and FAKE_OPENAI_KEY not in config.model_dump_json()
    record = run_one(FakeClient([bad]))
    assert FAKE_OPENAI_KEY not in record.model_dump_json()

    dataset, sha = ar.load_dataset(write_dataset(tmp_path, make_dataset()))
    plan = ar.build_run_plan(dataset, False)
    manifest = ar.build_manifest(run_id="r", started_at="t", config=config, dataset=dataset, dataset_sha256=sha,
                                 input_path=tmp_path / "s.json", plan=plan, summary=ar.RunSummary(), allow_unreviewed_demo=False)
    assert FAKE_OPENAI_KEY not in json.dumps(manifest)


def test_config_errors_do_not_echo_the_key():
    with pytest.raises(ar.ConfigError) as info:
        ar.load_config({"TARGET_PROVIDER": "openai", "TARGET_MODEL": "m", "OPENAI_API_KEY": FAKE_OPENAI_KEY,
                        "TARGET_TEMPERATURE": "9"})
    assert FAKE_OPENAI_KEY not in str(info.value)


# ---------------------------------------------------------------- config
def test_config_happy_path_and_defaults():
    cfg = ar.load_config({"TARGET_PROVIDER": "OpenAI", "TARGET_MODEL": "m", "OPENAI_API_KEY": FAKE_OPENAI_KEY})
    assert (cfg.target_provider, cfg.max_retries, cfg.timeout_seconds, cfg.temperature) == ("openai", 3, 60.0, 0.0)


def test_config_anthropic_ignores_temperature():
    cfg = ar.load_config({"TARGET_PROVIDER": "anthropic", "TARGET_MODEL": "m", "ANTHROPIC_API_KEY": "sk-ant-abcdefgh1234"})
    assert cfg.temperature is None


def test_config_empty_temperature_omits_it():
    cfg = ar.load_config({"TARGET_PROVIDER": "openai", "TARGET_MODEL": "m", "OPENAI_API_KEY": FAKE_OPENAI_KEY, "TARGET_TEMPERATURE": ""})
    assert cfg.temperature is None


@pytest.mark.parametrize("env, fragment", [
    ({}, "TARGET_PROVIDER is not set"),
    ({"TARGET_PROVIDER": "gemini"}, "Unsupported"),
    ({"TARGET_PROVIDER": "openai", "TARGET_MODEL": "m"}, "OPENAI_API_KEY is not set"),
    ({"TARGET_PROVIDER": "anthropic", "TARGET_MODEL": "m", "OPENAI_API_KEY": "sk-abcdefgh1234"}, "ANTHROPIC_API_KEY is not set"),
    ({"TARGET_PROVIDER": "openai", "OPENAI_API_KEY": "sk-abcdefgh1234"}, "TARGET_MODEL is not set"),
    ({"TARGET_PROVIDER": "openai", "TARGET_MODEL": "m", "OPENAI_API_KEY": "REPLACE_WITH_YOUR_OPENAI_API_KEY"}, "placeholder"),
    ({"TARGET_PROVIDER": "openai", "TARGET_MODEL": "m", "OPENAI_API_KEY": "sk-abcdefgh1234", "API_MAX_RETRIES": "50"}, "API_MAX_RETRIES"),
    ({"TARGET_PROVIDER": "openai", "TARGET_MODEL": "m", "OPENAI_API_KEY": "sk-abcdefgh1234", "API_TIMEOUT_SECONDS": "abc"}, "numeric"),
    ({"TARGET_PROVIDER": "openai", "TARGET_MODEL": "m", "OPENAI_API_KEY": "sk-abcdefgh1234", "API_TIMEOUT_SECONDS": "nan"}, "API_TIMEOUT_SECONDS"),
])
def test_config_failures_are_actionable(env, fragment):
    with pytest.raises(ar.ConfigError) as info:
        ar.load_config(env)
    assert fragment in str(info.value)


# ---------------------------------------------------------------- run loop
def test_run_continues_past_one_failure_and_counts_each_status(tmp_path):
    plan = ar.build_run_plan(ar.RunnerDataset.model_validate(make_dataset()), False)  # 2 calls
    records = []
    client = FakeClient([ar.ProviderError("bad_request", "x"), good_output()])
    summary = ar.run_evaluation(plan, client, make_config(), "r", ar.RetryPolicy(max_retries=0), records.append,
                                sleep=lambda s: None)
    assert summary.as_dict() == {"planned": 2, "attempted": 2, "success": 1, "api_error": 1, "invalid_response": 0}
    assert [r.status for r in records] == ["api_error", "success"]


def test_fatal_error_aborts_run():
    plan = ar.build_run_plan(ar.RunnerDataset.model_validate(make_dataset()), False)
    records = []
    client = FakeClient([ar.ProviderError("authentication", "bad key")])
    with pytest.raises(ar.RunAborted):
        ar.run_evaluation(plan, client, make_config(), "r", ar.RetryPolicy(max_retries=2), records.append, sleep=lambda s: None)
    assert len(records) == 1 and len(client.requests) == 1  # not retried, not continued


def test_consecutive_failure_streak_aborts_run():
    plan = ar.build_run_plan(ar.RunnerDataset.model_validate(make_dataset(*[make_scenario(f"S{i:02d}") for i in range(1, 5)])), False)
    client = FakeClient([good_output(confidence=3.0)])  # always invalid
    records = []
    with pytest.raises(ar.RunAborted, match="consecutive"):
        ar.run_evaluation(plan, client, make_config(), "r", ar.RetryPolicy(max_retries=0), records.append,
                          max_consecutive_failures=3, sleep=lambda s: None)
    assert len(records) == 3


# ---------------------------------------------------------------- CLI / file safety
def cli_env(monkeypatch):
    monkeypatch.setenv("TARGET_PROVIDER", "openai")
    monkeypatch.setenv("TARGET_MODEL", "target-x")
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_OPENAI_KEY)
    monkeypatch.setattr(ar, "load_dotenv", lambda *a, **k: None)


def test_existing_output_is_protected_before_any_api_call(tmp_path, monkeypatch, capsys):
    cli_env(monkeypatch)
    out = tmp_path / "agent_logs.jsonl"
    out.write_text("precious paid results\n", encoding="utf-8")
    created = []
    code = ar.main(["--input", str(write_dataset(tmp_path, make_dataset())), "--output", str(out), "--yes"],
                   client_factory=lambda cfg: created.append(cfg) or FakeClient([good_output()]))
    assert code == 1 and created == []
    assert out.read_text(encoding="utf-8") == "precious paid results\n"
    assert "--force" in capsys.readouterr().err


def test_stale_partial_file_also_blocks_a_run(tmp_path, monkeypatch):
    cli_env(monkeypatch)
    out = tmp_path / "agent_logs.jsonl"
    ar.partial_path(out).write_text("leftover\n", encoding="utf-8")
    code = ar.main(["--input", str(write_dataset(tmp_path, make_dataset())), "--output", str(out), "--yes"],
                   client_factory=lambda cfg: FakeClient([good_output()]))
    assert code == 1 and not out.exists()


def test_full_run_writes_valid_jsonl_and_manifest_atomically(tmp_path, monkeypatch, capsys):
    cli_env(monkeypatch)
    out = tmp_path / "agent_logs.jsonl"
    client = FakeClient([good_output(), good_output(decision=OPTIONS[0], confidence=0.4)])
    code = ar.main(["--input", str(write_dataset(tmp_path, make_dataset())), "--output", str(out), "--yes"],
                   client_factory=lambda cfg: client, sleep=lambda s: None)
    assert code == 0
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert [l["variant_id"] for l in lines] == ["S01-BASE", "S01-V1"]
    assert all(l["status"] == "success" and l["provider"] == "openai" and l["model"] == "target-x" for l in lines)
    assert not ar.partial_path(out).exists()
    manifest = json.loads(ar.manifest_path(out).read_text(encoding="utf-8"))
    assert manifest["counts"]["success"] == 2 and manifest["provisional"] is False
    assert manifest["target_model_equals_generator_model"] is False
    assert FAKE_OPENAI_KEY not in out.read_text() + ar.manifest_path(out).read_text()
    stdout = capsys.readouterr().out
    assert "COST NOTICE" in stdout and "No metrics were computed" in stdout and FAKE_OPENAI_KEY not in stdout


def test_force_replaces_output_but_keeps_backup(tmp_path, monkeypatch):
    cli_env(monkeypatch)
    out = tmp_path / "agent_logs.jsonl"
    out.write_text("old\n", encoding="utf-8")
    code = ar.main(["--input", str(write_dataset(tmp_path, make_dataset())), "--output", str(out), "--yes", "--force"],
                   client_factory=lambda cfg: FakeClient([good_output()]), sleep=lambda s: None)
    assert code == 0
    assert "old" not in out.read_text(encoding="utf-8")
    assert any(p.name.startswith("agent_logs.jsonl.bak-") for p in tmp_path.iterdir())


def test_aborted_run_leaves_only_a_partial_file(tmp_path, monkeypatch, capsys):
    cli_env(monkeypatch)
    out = tmp_path / "agent_logs.jsonl"
    code = ar.main(["--input", str(write_dataset(tmp_path, make_dataset())), "--output", str(out), "--yes"],
                   client_factory=lambda cfg: FakeClient([ar.ProviderError("authentication", "x")]), sleep=lambda s: None)
    assert code == 1
    assert not out.exists() and not ar.manifest_path(out).exists()
    assert ar.partial_path(out).exists()
    assert "NOT a completed run" in capsys.readouterr().err


def test_keyboard_interrupt_leaves_partial_only(tmp_path, monkeypatch):
    cli_env(monkeypatch)
    out = tmp_path / "agent_logs.jsonl"

    class Interrupting(FakeClient):
        def complete(self, request):
            if self.requests:
                raise KeyboardInterrupt
            return super().complete(request)

    code = ar.main(["--input", str(write_dataset(tmp_path, make_dataset())), "--output", str(out), "--yes"],
                   client_factory=lambda cfg: Interrupting([good_output()]), sleep=lambda s: None)
    assert code == 130 and not out.exists() and ar.partial_path(out).exists()
    assert len(ar.partial_path(out).read_text(encoding="utf-8").splitlines()) == 1


def test_unapproved_dataset_is_refused_by_default_and_runs_provisionally_in_demo(tmp_path, monkeypatch, capsys):
    cli_env(monkeypatch)
    data = make_dataset(make_scenario("S01", review="pending_review"))
    inp = write_dataset(tmp_path, data)
    out = tmp_path / "agent_logs.jsonl"
    created = []
    assert ar.main(["--input", str(inp), "--output", str(out), "--yes"],
                   client_factory=lambda cfg: created.append(1) or FakeClient([good_output()])) == 1
    assert created == []
    assert ar.main(["--input", str(inp), "--output", str(out), "--yes", "--allow-unreviewed-demo"],
                   client_factory=lambda cfg: FakeClient([good_output()]), sleep=lambda s: None) == 0
    records = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert all(r["provisional"] is True for r in records)
    assert json.loads(ar.manifest_path(out).read_text())["provisional"] is True
    assert "PROVISIONAL" in capsys.readouterr().out


def test_non_interactive_session_requires_yes(tmp_path, monkeypatch, capsys):
    cli_env(monkeypatch)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    created = []
    code = ar.main(["--input", str(write_dataset(tmp_path, make_dataset())), "--output", str(tmp_path / "o.jsonl")],
                   client_factory=lambda cfg: created.append(1) or FakeClient([good_output()]))
    assert code == 1 and created == [] and "--yes" in capsys.readouterr().err


def test_declining_confirmation_makes_no_api_calls(tmp_path, monkeypatch):
    cli_env(monkeypatch)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    created = []
    code = ar.main(["--input", str(write_dataset(tmp_path, make_dataset())), "--output", str(tmp_path / "o.jsonl")],
                   client_factory=lambda cfg: created.append(1) or FakeClient([good_output()]), input_fn=lambda prompt: "no")
    assert code == 1 and created == []
