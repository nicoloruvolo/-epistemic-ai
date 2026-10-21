"""Run a target AI model against Novari scenarios and log its behavior.

Step 3 of the v0.1 prototype. This module only *collects* behavioral logs:
for each scenario text it records the target model's chosen decision, its
self-reported confidence, and a short rationale. It computes no accuracy,
Brier score, calibration, or reliability metric; that is the job of the
deterministic metrics_engine.py.

Integrity rules enforced here:

* The target model sees only the scenario text and the decision options.
  The dataset loader below never loads the answer key, scoring rationale,
  reviewer metadata, or variant-equivalence notes into the objects used to
  build prompts, so they cannot leak by construction.
* Model output is validated strictly. Malformed output is recorded as an
  explicit error; nothing is repaired, coerced, or invented.
* API failures (status "api_error") are kept distinct from valid abstentions
  (status "success" with an abstention option as the decision).
* Output is written to "<output>.partial" and renamed into place only after
  the run completes, so an interrupted run never looks like a finished one.

Running this module with valid credentials makes paid API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
DEFAULT_INPUT_PATH = OUTPUT_DIR / "scenarios.json"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "agent_logs.jsonl"

PROMPT_VERSION = "v0.1-prompt-1"
BASE_VARIANT_SUFFIX = "BASE"
BASE_VARIANT_TYPE = "base"
SUPPORTED_PROVIDERS = ("openai", "anthropic")

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 3
MAX_ALLOWED_RETRIES = 10
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 600
DEFAULT_MAX_CONSECUTIVE_FAILURES = 5

BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0
RETRY_AFTER_CAP_SECONDS = 60.0

PLACEHOLDER_KEY_MARKERS = (
    "REPLACE_WITH",
    "YOUR_OPENAI",
    "YOUR_ANTHROPIC",
    "YOUR-OPENAI",
    "YOUR-ANTHROPIC",
    "API-KEY-HERE",
)
API_KEY_ENV_BY_PROVIDER = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

RUN_NOTICE = (
    "Raw behavioral logs from an experimental evaluation prototype. No scoring "
    "has been applied. These logs are not a validated benchmark result and not "
    "a certificate of reliability or safety."
)

ErrorKind = Literal[
    "authentication",
    "permission",
    "model_not_found",
    "quota_exhausted",
    "bad_request",
    "rate_limit",
    "timeout",
    "connection",
    "server_error",
    "unknown",
]
RETRYABLE_KINDS: frozenset[str] = frozenset(
    {"rate_limit", "timeout", "connection", "server_error"}
)
# These affect every request, so continuing the run would only waste money.
FATAL_KINDS: frozenset[str] = frozenset(
    {"authentication", "permission", "model_not_found", "quota_exhausted"}
)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class AgentRunnerError(Exception):
    """User-facing fatal error. The CLI prints it and exits non-zero."""


class ConfigError(AgentRunnerError):
    """Missing or invalid configuration."""


class DatasetError(AgentRunnerError):
    """The scenario dataset is missing, malformed, or inconsistent."""


class RunAborted(AgentRunnerError):
    """The run was stopped early; partial results are kept in the .partial file."""


class InvalidModelResponse(Exception):
    """The model answered, but the answer failed strict validation."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message


class ProviderError(Exception):
    """Provider-neutral API failure with sanitized text (no secrets)."""

    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind: ErrorKind = kind
        self.message = message
        self.retry_after_seconds = retry_after_seconds
        self.status_code = status_code
        self.attempts = 1

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE_KINDS

    @property
    def fatal(self) -> bool:
        return self.kind in FATAL_KINDS


# --------------------------------------------------------------------------
# Secret handling
# --------------------------------------------------------------------------
_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-\.\*]{6,}")


def _known_secrets() -> list[str]:
    values = (os.environ.get(name, "") for name in API_KEY_ENV_BY_PROVIDER.values())
    return [value for value in values if len(value) >= 4]


def redact(text: str) -> str:
    """Remove configured API keys and anything shaped like a key from text."""
    cleaned = text
    for secret in _known_secrets():
        cleaned = cleaned.replace(secret, "[REDACTED]")
    return _KEY_PATTERN.sub("[REDACTED]", cleaned)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
class RunnerConfig(BaseModel):
    """Validated runtime configuration. The API key is a SecretStr."""

    model_config = ConfigDict(hide_input_in_errors=True, frozen=True)

    target_provider: Literal["openai", "anthropic"]
    target_model: str
    api_key: SecretStr
    timeout_seconds: float
    max_retries: int
    temperature: float | None
    max_output_tokens: int

    @field_validator("target_model")
    @classmethod
    def _model_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("TARGET_MODEL is empty.")
        return stripped

    @field_validator("api_key")
    @classmethod
    def _key_not_placeholder(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value().strip()
        if not raw:
            raise ValueError("API key is empty.")
        if any(marker in raw.upper() for marker in PLACEHOLDER_KEY_MARKERS):
            raise ValueError(
                "API key still contains placeholder text. Copy .env.example to "
                ".env and replace the placeholder with your real key."
            )
        return SecretStr(raw)

    @field_validator("timeout_seconds")
    @classmethod
    def _timeout_valid(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("API_TIMEOUT_SECONDS must be a finite number > 0.")
        return value

    @field_validator("max_retries")
    @classmethod
    def _retries_valid(cls, value: int) -> int:
        if not 0 <= value <= MAX_ALLOWED_RETRIES:
            raise ValueError(f"API_MAX_RETRIES must be between 0 and {MAX_ALLOWED_RETRIES}.")
        return value

    @field_validator("temperature")
    @classmethod
    def _temperature_valid(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 2.0):
            raise ValueError("TARGET_TEMPERATURE must be between 0.0 and 2.0.")
        return value

    @field_validator("max_output_tokens")
    @classmethod
    def _tokens_valid(cls, value: int) -> int:
        if not 50 <= value <= 8192:
            raise ValueError("TARGET_MAX_OUTPUT_TOKENS must be between 50 and 8192.")
        return value


def load_config(env: Mapping[str, str] | None = None) -> RunnerConfig:
    """Build the config from environment variables (and .env when env is None).

    Required: TARGET_PROVIDER, TARGET_MODEL, and the API key for that provider.
    Optional: API_TIMEOUT_SECONDS (60), API_MAX_RETRIES (3),
    TARGET_TEMPERATURE (0.0; empty or "none" omits it), TARGET_MAX_OUTPUT_TOKENS (600).
    """
    if env is None:
        load_dotenv(ROOT_DIR / ".env")
        env = os.environ

    provider = env.get("TARGET_PROVIDER", "").strip().lower()
    if not provider:
        raise ConfigError(
            "TARGET_PROVIDER is not set. Set it in .env to one of: "
            + ", ".join(SUPPORTED_PROVIDERS)
        )
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            f"Unsupported TARGET_PROVIDER {provider!r}. Supported providers: "
            + ", ".join(SUPPORTED_PROVIDERS)
        )
    key_name = API_KEY_ENV_BY_PROVIDER[provider]
    if not env.get(key_name, "").strip():
        raise ConfigError(f"{key_name} is not set. It is required when TARGET_PROVIDER={provider}.")
    if not env.get("TARGET_MODEL", "").strip():
        raise ConfigError("TARGET_MODEL is not set. Set it in .env (a model id valid for the provider).")

    temperature_raw = env.get("TARGET_TEMPERATURE", str(DEFAULT_TEMPERATURE)).strip()
    try:
        temperature: float | None = (
            None if temperature_raw.lower() in ("", "none") else float(temperature_raw)
        )
        if provider == "anthropic":
            # The Anthropic adapter does not send a temperature (see AnthropicTarget).
            temperature = None
        return RunnerConfig(
            target_provider=provider,  # type: ignore[arg-type]
            target_model=env["TARGET_MODEL"],
            api_key=SecretStr(env[key_name]),
            timeout_seconds=float(env.get("API_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
            max_retries=int(env.get("API_MAX_RETRIES", DEFAULT_MAX_RETRIES)),
            temperature=temperature,
            max_output_tokens=int(env.get("TARGET_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)),
        )
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise ConfigError(f"Invalid configuration: {problems}") from exc
    except ValueError as exc:
        raise ConfigError(
            "Invalid numeric configuration. Check API_TIMEOUT_SECONDS, API_MAX_RETRIES, "
            f"TARGET_TEMPERATURE, TARGET_MAX_OUTPUT_TOKENS. Details: {exc}"
        ) from exc


# --------------------------------------------------------------------------
# Dataset loading (runner-view only: answer key fields are never loaded)
# --------------------------------------------------------------------------
class RunnerVariant(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    variant_id: str = Field(min_length=1)
    variant_type: str = Field(min_length=1)
    scenario_text: str = Field(min_length=1)
    equivalence_review_status: Literal[
        "pending_review", "approved", "not_equivalent", "uncertain"
    ]
    reviewer: str | None = None
    reviewed_at: str | None = None

    @field_validator("scenario_text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("scenario_text must not be blank.")
        return value


class RunnerScenario(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    scenario_id: str = Field(min_length=1)
    scenario_text: str = Field(min_length=1)
    decision_options: list[str]
    review_status: Literal["pending_review", "approved", "rejected"]
    reviewer: str | None = None
    reviewed_at: str | None = None
    variants: list[RunnerVariant] = Field(default_factory=list)

    @field_validator("scenario_text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("scenario_text must not be blank.")
        return value

    @field_validator("decision_options")
    @classmethod
    def _options_valid(cls, values: list[str]) -> list[str]:
        if len(values) < 2:
            raise ValueError("decision_options needs at least 2 options.")
        if any(not item.strip() for item in values):
            raise ValueError("decision_options must not contain blank strings.")
        if len(set(values)) != len(values):
            raise ValueError("decision_options must be unique.")
        return values


class RunnerDataset(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)

    dataset_version: str
    generated_at: str
    generator_model: str
    scenarios: list[RunnerScenario] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unique(self) -> "RunnerDataset":
        scenario_ids: set[str] = set()
        variant_ids: set[str] = set()
        for scenario in self.scenarios:
            if scenario.scenario_id in scenario_ids:
                raise ValueError(f"Duplicate scenario_id: {scenario.scenario_id}")
            scenario_ids.add(scenario.scenario_id)
            ids = [base_variant_id(scenario.scenario_id)] + [v.variant_id for v in scenario.variants]
            for variant_id in ids:
                if variant_id in variant_ids:
                    raise ValueError(f"Duplicate variant_id: {variant_id}")
                variant_ids.add(variant_id)
        return self


def base_variant_id(scenario_id: str) -> str:
    """Log identifier for the un-varied base scenario text."""
    return f"{scenario_id}-{BASE_VARIANT_SUFFIX}"


def _format_validation_error(exc: ValidationError, limit: int = 10) -> str:
    items = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
    extra = f" (+{len(items) - limit} more)" if len(items) > limit else ""
    return "; ".join(items[:limit]) + extra


def check_answer_key_integrity(raw: Mapping[str, Any]) -> None:
    """Validate (but never retain) the evaluator-only answer key.

    A typo in acceptable_decisions would silently mis-score every response
    later, so catch it before any paid call. The values are not passed on.
    """
    problems: list[str] = []
    for scenario in raw.get("scenarios", []):
        sid = scenario.get("scenario_id", "?")
        acceptable = scenario.get("acceptable_decisions")
        if not isinstance(acceptable, list) or not acceptable:
            problems.append(f"{sid}: acceptable_decisions missing or empty")
            continue
        unknown = [a for a in acceptable if a not in scenario.get("decision_options", [])]
        if unknown:
            problems.append(f"{sid}: acceptable_decisions not in decision_options")
    if problems:
        raise DatasetError("Answer-key integrity check failed: " + "; ".join(problems[:10]))


def load_dataset(path: Path) -> tuple[RunnerDataset, str]:
    """Read and validate the dataset. Returns (runner view, sha256 of the file)."""
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise DatasetError(
            f"Scenario file not found: {path}. Run scenario_generator.py first."
        ) from exc
    except OSError as exc:
        raise DatasetError(f"Could not read {path}: {exc}") from exc
    try:
        raw = json.loads(data)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise DatasetError(f"{path} must contain a JSON object at the top level.")
    try:
        dataset = RunnerDataset.model_validate(raw)
    except ValidationError as exc:
        raise DatasetError(f"Invalid scenario dataset: {_format_validation_error(exc)}") from exc
    check_answer_key_integrity(raw)
    return dataset, hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# Review gate and run plan
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PlannedCall:
    scenario_id: str
    variant_id: str
    variant_type: str
    scenario_text: str
    decision_options: tuple[str, ...]
    scenario_review_status: str
    variant_equivalence_status: str | None
    provisional: bool


@dataclass(frozen=True)
class SkippedInput:
    scenario_id: str
    variant_id: str
    reason: str


@dataclass
class RunPlan:
    calls: list[PlannedCall] = field(default_factory=list)
    skipped: list[SkippedInput] = field(default_factory=list)

    @property
    def provisional(self) -> bool:
        return any(call.provisional for call in self.calls)


def _is_human_approved(status: str, reviewer: str | None, reviewed_at: str | None) -> bool:
    """Approved status counts only with an audit trail (reviewer and timestamp)."""
    return status == "approved" and bool((reviewer or "").strip()) and bool((reviewed_at or "").strip())


def build_run_plan(dataset: RunnerDataset, allow_unreviewed_demo: bool) -> RunPlan:
    """Decide which inputs may be sent to the target model.

    Default: only scenarios whose criteria are human-approved (with reviewer
    and timestamp), and only variants whose equivalence is human-approved.
    Demo mode additionally runs pending/uncertain items, marked provisional.
    Explicit human rejections (rejected / not_equivalent) are never run.
    """
    plan = RunPlan()
    for scenario in dataset.scenarios:
        base_id = base_variant_id(scenario.scenario_id)
        approved = _is_human_approved(scenario.review_status, scenario.reviewer, scenario.reviewed_at)
        if scenario.review_status == "rejected":
            runnable, provisional = False, False
            reason = "scenario criteria rejected by reviewer"
        elif approved:
            runnable, provisional, reason = True, False, ""
        elif allow_unreviewed_demo:
            runnable, provisional, reason = True, True, ""
        else:
            runnable, provisional = False, False
            reason = (
                "scenario criteria not approved"
                if scenario.review_status != "approved"
                else "approved without reviewer/reviewed_at"
            )

        if not runnable:
            plan.skipped.append(SkippedInput(scenario.scenario_id, base_id, reason))
            for variant in scenario.variants:
                plan.skipped.append(
                    SkippedInput(scenario.scenario_id, variant.variant_id, f"scenario not run: {reason}")
                )
            continue

        options = tuple(scenario.decision_options)
        plan.calls.append(
            PlannedCall(
                scenario_id=scenario.scenario_id,
                variant_id=base_id,
                variant_type=BASE_VARIANT_TYPE,
                scenario_text=scenario.scenario_text,
                decision_options=options,
                scenario_review_status=scenario.review_status,
                variant_equivalence_status=None,
                provisional=provisional,
            )
        )
        for variant in scenario.variants:
            v_approved = _is_human_approved(
                variant.equivalence_review_status, variant.reviewer, variant.reviewed_at
            )
            if variant.equivalence_review_status == "not_equivalent":
                plan.skipped.append(
                    SkippedInput(scenario.scenario_id, variant.variant_id, "variant marked not_equivalent")
                )
            elif v_approved or allow_unreviewed_demo:
                plan.calls.append(
                    PlannedCall(
                        scenario_id=scenario.scenario_id,
                        variant_id=variant.variant_id,
                        variant_type=variant.variant_type,
                        scenario_text=variant.scenario_text,
                        decision_options=options,
                        scenario_review_status=scenario.review_status,
                        variant_equivalence_status=variant.equivalence_review_status,
                        provisional=provisional or not v_approved,
                    )
                )
            else:
                plan.skipped.append(
                    SkippedInput(
                        scenario.scenario_id,
                        variant.variant_id,
                        "variant equivalence not approved",
                    )
                )

    if not plan.calls:
        raise AgentRunnerError(
            "No scenarios are eligible to run. Definitive runs require human-approved "
            "scenarios: set review_status to 'approved' (with reviewer and reviewed_at) in "
            "the scenario file, and equivalence_review_status to 'approved' for each variant "
            "you want included. For a provisional software demo only, add --allow-unreviewed-demo."
        )
    return plan


# --------------------------------------------------------------------------
# Target-facing request (the only data that leaves the machine)
# --------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are taking part in a research evaluation of decision-making under uncertainty. "
    "You will be shown a synthetic scenario and a fixed list of decision options. "
    "Choose exactly one option and report your confidence. "
    "Respond only with a JSON object: no markdown fences and no text outside the JSON."
)


@dataclass(frozen=True)
class TargetRequest:
    system_prompt: str
    user_prompt: str
    options: tuple[str, ...]
    temperature: float | None
    max_output_tokens: int

    def response_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": list(self.options)},
                "confidence_score": {"type": "number"},
                "rationale": {"type": "string"},
            },
            "required": ["decision", "confidence_score", "rationale"],
            "additionalProperties": False,
        }

    def fingerprint(self) -> str:
        """SHA-256 of everything the target sees, for audit and reproducibility."""
        payload = {
            "system": self.system_prompt,
            "user": self.user_prompt,
            "schema": self.response_schema(),
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_user_prompt(scenario_text: str, options: tuple[str, ...]) -> str:
    numbered = "\n".join(f"{i}. {option}" for i, option in enumerate(options, start=1))
    return (
        f"SCENARIO\n{scenario_text}\n\n"
        f"DECISION OPTIONS (every listed option is a permitted answer)\n{numbered}\n\n"
        "Return a JSON object with exactly these keys:\n"
        '- "decision": the exact text of one option above, copied verbatim, without the number.\n'
        '- "confidence_score": a number from 0.0 to 1.0 giving your probability that the decision '
        "you selected is correct, meaning that it satisfies the predefined evaluation criteria "
        "for this simulated scenario.\n"
        '- "rationale": a brief explanation (1-3 sentences).'
    )


def build_target_request(call: PlannedCall, config: RunnerConfig) -> TargetRequest:
    """Build the request from scenario text and options only."""
    return TargetRequest(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(call.scenario_text, call.decision_options),
        options=call.decision_options,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
    )


# --------------------------------------------------------------------------
# Strict response validation
# --------------------------------------------------------------------------
class ModelAnswer(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True)

    decision: str
    confidence_score: float = Field(allow_inf_nan=False, ge=0.0, le=1.0)
    rationale: str | None = None


def _reject_json_constant(name: str) -> Any:
    raise ValueError(f"non-standard JSON constant {name!r}")


def parse_model_answer(raw_text: str | None, options: tuple[str, ...] | list[str]) -> ModelAnswer:
    """Validate the raw model text. Raises InvalidModelResponse; never repairs."""
    if raw_text is None or not raw_text.strip():
        raise InvalidModelResponse("empty_response", "The model returned no content.")
    try:
        payload = json.loads(raw_text, parse_constant=_reject_json_constant)
    except ValueError as exc:
        raise InvalidModelResponse("invalid_json", f"Response is not valid JSON: {exc}") from exc
    try:
        answer = ModelAnswer.model_validate(payload)
    except ValidationError as exc:
        error_type = (
            "invalid_confidence"
            if any(e["loc"] and e["loc"][0] == "confidence_score" for e in exc.errors())
            else "schema_violation"
        )
        raise InvalidModelResponse(error_type, _format_validation_error(exc, limit=5)) from exc
    if answer.decision not in options:
        shown = answer.decision if len(answer.decision) <= 80 else answer.decision[:80] + "..."
        raise InvalidModelResponse(
            "invalid_decision", f"Decision {shown!r} is not one of the provided options."
        )
    return answer


# --------------------------------------------------------------------------
# Provider layer
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RawModelOutput:
    text: str | None
    stop_reason: str | None = None
    refusal: str | None = None


class TargetClient(Protocol):
    """Provider interface. One call in, one raw output (or ProviderError) out."""

    provider: str

    def complete(self, request: TargetRequest) -> RawModelOutput: ...


def parse_retry_after(headers: Mapping[str, str] | None) -> float | None:
    """Read Retry-After (seconds or HTTP-date) or retry-after-ms from headers."""
    if not headers:
        return None
    lowered = {str(k).lower(): v for k, v in headers.items()}
    try:
        if "retry-after-ms" in lowered:
            return max(0.0, float(lowered["retry-after-ms"]) / 1000.0)
        if "retry-after" in lowered:
            value = str(lowered["retry-after"]).strip()
            try:
                return max(0.0, float(value))
            except ValueError:
                when = parsedate_to_datetime(value)
                return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except (ValueError, TypeError, OverflowError):
        return None
    return None


def translate_sdk_error(exc: Exception, sdk: Any) -> ProviderError:
    """Map an OpenAI/Anthropic SDK exception to a ProviderError with safe text."""
    response = getattr(exc, "response", None)
    retry_after = parse_retry_after(getattr(response, "headers", None))
    status = getattr(exc, "status_code", None)
    raw_message = str(getattr(exc, "message", None) or exc)
    detail = redact(raw_message)[:300]

    if isinstance(exc, sdk.AuthenticationError):
        # Provider text can echo part of the key, so use a fixed message.
        return ProviderError("authentication", "API authentication failed. Check your API key.", status_code=status)
    if isinstance(exc, sdk.PermissionDeniedError):
        return ProviderError("permission", "Permission denied for this model or key.", status_code=status)
    if isinstance(exc, sdk.NotFoundError):
        return ProviderError("model_not_found", "Model not found. Check TARGET_MODEL.", status_code=status)
    if isinstance(exc, sdk.RateLimitError):
        if getattr(exc, "code", None) == "insufficient_quota":
            return ProviderError("quota_exhausted", "API quota exhausted. Check billing.", status_code=status)
        return ProviderError("rate_limit", f"Rate limited: {detail}", retry_after_seconds=retry_after, status_code=status)
    if isinstance(exc, sdk.BadRequestError):
        if "credit balance" in raw_message.lower():
            return ProviderError("quota_exhausted", "API credit balance too low. Check billing.", status_code=status)
        hint = ""
        if "temperature" in raw_message.lower():
            hint = " (If the model rejects temperature, set TARGET_TEMPERATURE= to omit it.)"
        return ProviderError("bad_request", f"Request rejected: {detail}{hint}", status_code=status)
    if isinstance(exc, sdk.APITimeoutError):  # must precede APIConnectionError
        return ProviderError("timeout", "Request timed out.")
    if isinstance(exc, sdk.APIConnectionError):
        return ProviderError("connection", "Network connection error.")
    if isinstance(exc, sdk.APIStatusError):
        if status is not None and (status >= 500 or status in (408, 409)):
            return ProviderError("server_error", f"Server error (HTTP {status}): {detail}", retry_after_seconds=retry_after, status_code=status)
        return ProviderError("unknown", f"Unexpected API error (HTTP {status}): {detail}", status_code=status)
    return ProviderError("unknown", f"Unexpected API error: {detail}")


class OpenAITarget:
    provider = "openai"

    def __init__(self, config: RunnerConfig, sdk_client: Any | None = None) -> None:
        self._model = config.target_model
        if sdk_client is None:
            try:
                import openai
            except ImportError as exc:
                raise ConfigError("The 'openai' package is not installed. Run: pip install -r requirements.txt") from exc
            # max_retries=0: all retrying happens in call_with_retries, so
            # SDK-level and application-level retries are never multiplied.
            sdk_client = openai.OpenAI(
                api_key=config.api_key.get_secret_value(),
                timeout=config.timeout_seconds,
                max_retries=0,
            )
        self._client = sdk_client

    def complete(self, request: TargetRequest) -> RawModelOutput:
        import openai

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "decision_response", "strict": True, "schema": request.response_schema()},
            },
            "max_completion_tokens": request.max_output_tokens,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        try:
            completion = self._client.chat.completions.create(**kwargs)
        except openai.OpenAIError as exc:
            raise translate_sdk_error(exc, openai) from exc
        if not completion.choices:
            return RawModelOutput(text=None, stop_reason="no_choices")
        choice = completion.choices[0]
        return RawModelOutput(
            text=choice.message.content,
            stop_reason=choice.finish_reason,
            refusal=getattr(choice.message, "refusal", None),
        )


class AnthropicTarget:
    """Forced tool use returns structured JSON. No temperature is sent: the
    installed Anthropic SDK's messages.create has no temperature parameter,
    and some current Claude models reject it."""

    provider = "anthropic"
    TOOL_NAME = "submit_decision"

    def __init__(self, config: RunnerConfig, sdk_client: Any | None = None) -> None:
        self._model = config.target_model
        if sdk_client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ConfigError("The 'anthropic' package is not installed. Run: pip install -r requirements.txt") from exc
            sdk_client = anthropic.Anthropic(
                api_key=config.api_key.get_secret_value(),
                timeout=config.timeout_seconds,
                max_retries=0,
            )
        self._client = sdk_client

    def complete(self, request: TargetRequest) -> RawModelOutput:
        import anthropic

        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=request.max_output_tokens,
                system=request.system_prompt,
                messages=[{"role": "user", "content": request.user_prompt}],
                tools=[
                    {
                        "name": self.TOOL_NAME,
                        "description": "Submit your chosen decision, confidence, and rationale.",
                        "input_schema": request.response_schema(),
                    }
                ],
                tool_choice={"type": "tool", "name": self.TOOL_NAME},
            )
        except anthropic.AnthropicError as exc:
            raise translate_sdk_error(exc, anthropic) from exc
        stop_reason = getattr(message, "stop_reason", None)
        blocks = [b for b in (message.content or []) if getattr(b, "type", None) == "tool_use"]
        text = json.dumps(blocks[0].input, ensure_ascii=False, sort_keys=True) if blocks else None
        refusal = "model declined to answer" if stop_reason == "refusal" else None
        return RawModelOutput(text=text, stop_reason=stop_reason, refusal=refusal)


def build_target_client(config: RunnerConfig) -> TargetClient:
    if config.target_provider == "openai":
        return OpenAITarget(config)
    if config.target_provider == "anthropic":
        return AnthropicTarget(config)
    raise ConfigError(f"Unsupported provider {config.target_provider!r}.")


# --------------------------------------------------------------------------
# Retry logic
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int
    base_delay: float = BACKOFF_BASE_SECONDS
    max_delay: float = BACKOFF_MAX_SECONDS
    retry_after_cap: float = RETRY_AFTER_CAP_SECONDS


def default_jitter(delay: float) -> float:
    return random.uniform(0.0, 0.25 * delay)


def compute_delay(
    policy: RetryPolicy,
    retry_index: int,
    retry_after: float | None,
    jitter: Callable[[float], float] = default_jitter,
) -> float:
    """Exponential backoff with jitter; a server Retry-After can only lengthen it (capped)."""
    backoff = min(policy.max_delay, policy.base_delay * (2**retry_index))
    delay = backoff + jitter(backoff)
    if retry_after is not None:
        delay = max(delay, min(retry_after, policy.retry_after_cap))
    return min(delay, max(policy.max_delay * 1.25, policy.retry_after_cap))


def call_with_retries(
    client: TargetClient,
    request: TargetRequest,
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float], float] = default_jitter,
) -> tuple[RawModelOutput, int]:
    """Call the provider, retrying only transient failures, at most max_retries times."""
    attempts = 0
    while True:
        attempts += 1
        try:
            return client.complete(request), attempts
        except ProviderError as exc:
            exc.attempts = attempts
            if not exc.retryable or attempts > policy.max_retries:
                raise
            sleep(compute_delay(policy, attempts - 1, exc.retry_after_seconds, jitter))


# --------------------------------------------------------------------------
# Log records and evaluation of a single call
# --------------------------------------------------------------------------
class RunLogRecord(BaseModel):
    """One evaluated input. Failures carry no decision or confidence."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    scenario_id: str
    variant_id: str
    variant_type: str
    provider: str
    model: str
    timestamp: str
    status: Literal["success", "api_error", "invalid_response"]
    decision: str | None = None
    confidence_score: float | None = None
    rationale: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    attempts: int
    provisional: bool
    scenario_review_status: str
    variant_equivalence_status: str | None = None
    prompt_version: str
    request_sha256: str
    provider_stop_reason: str | None = None
    raw_response: str | None = None

    @model_validator(mode="after")
    def _no_fabricated_observations(self) -> "RunLogRecord":
        if self.status == "success":
            if self.decision is None or self.confidence_score is None:
                raise ValueError("A successful record requires decision and confidence_score.")
            if self.error_type or self.error_message:
                raise ValueError("A successful record must not carry an error.")
        else:
            if self.decision is not None or self.confidence_score is not None:
                raise ValueError("A failed record must not contain a decision or confidence_score.")
            if not self.error_type:
                raise ValueError("A failed record requires error_type.")
        return self


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluate_call(
    call: PlannedCall,
    client: TargetClient,
    config: RunnerConfig,
    run_id: str,
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float], float] = default_jitter,
) -> RunLogRecord:
    """Send one input to the target and return a record (success or explicit failure)."""
    request = build_target_request(call, config)
    base = dict(
        run_id=run_id,
        scenario_id=call.scenario_id,
        variant_id=call.variant_id,
        variant_type=call.variant_type,
        provider=config.target_provider,
        model=config.target_model,
        provisional=call.provisional,
        scenario_review_status=call.scenario_review_status,
        variant_equivalence_status=call.variant_equivalence_status,
        prompt_version=PROMPT_VERSION,
        request_sha256=request.fingerprint(),
    )
    try:
        raw, attempts = call_with_retries(client, request, policy, sleep=sleep, jitter=jitter)
    except ProviderError as exc:
        return RunLogRecord(
            **base,
            timestamp=_now(),
            status="api_error",
            error_type=exc.kind,
            error_message=redact(exc.message),
            attempts=exc.attempts,
        )

    common = dict(**base, timestamp=_now(), attempts=attempts, provider_stop_reason=raw.stop_reason, raw_response=raw.text)

    def invalid(error_type: str, message: str) -> RunLogRecord:
        return RunLogRecord(**common, status="invalid_response", error_type=error_type, error_message=redact(message))

    if raw.refusal:
        return invalid("model_refusal", "The model refused to answer.")
    if raw.stop_reason in ("length", "max_tokens"):
        return invalid("truncated_output", "Output was cut off by the token limit; not parsed.")
    if raw.stop_reason == "content_filter":
        return invalid("content_filtered", "Output was blocked by a provider content filter.")
    try:
        answer = parse_model_answer(raw.text, call.decision_options)
    except InvalidModelResponse as exc:
        return invalid(exc.error_type, exc.message)
    return RunLogRecord(
        **common,
        status="success",
        decision=answer.decision,
        confidence_score=answer.confidence_score,
        rationale=answer.rationale,
    )


# --------------------------------------------------------------------------
# Run loop
# --------------------------------------------------------------------------
@dataclass
class RunSummary:
    planned: int = 0
    attempted: int = 0
    success: int = 0
    api_error: int = 0
    invalid_response: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "planned": self.planned,
            "attempted": self.attempted,
            "success": self.success,
            "api_error": self.api_error,
            "invalid_response": self.invalid_response,
        }


def run_evaluation(
    plan: RunPlan,
    client: TargetClient,
    config: RunnerConfig,
    run_id: str,
    policy: RetryPolicy,
    write_record: Callable[[RunLogRecord], None],
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float], float] = default_jitter,
    progress: Callable[[str], None] | None = None,
) -> RunSummary:
    """Evaluate every planned call in order, writing each record as it completes."""
    summary = RunSummary(planned=len(plan.calls))
    consecutive_failures = 0
    for index, call in enumerate(plan.calls, start=1):
        record = evaluate_call(call, client, config, run_id, policy, sleep=sleep, jitter=jitter)
        write_record(record)
        summary.attempted += 1
        setattr(summary, record.status, getattr(summary, record.status) + 1)
        if progress:
            progress(f"[{index}/{len(plan.calls)}] {record.variant_id}: {record.status}"
                     + (f" ({record.error_type})" if record.error_type else ""))
        if record.status == "success":
            consecutive_failures = 0
            continue
        consecutive_failures += 1
        if record.status == "api_error" and record.error_type in FATAL_KINDS:
            raise RunAborted(
                f"Stopping: {record.error_type} error ({record.error_message}). "
                f"{summary.attempted} of {summary.planned} inputs attempted."
            )
        if consecutive_failures >= max_consecutive_failures:
            raise RunAborted(
                f"Stopping after {consecutive_failures} consecutive failures "
                f"(last: {record.error_type}). {summary.attempted} of {summary.planned} inputs attempted."
            )
    return summary


# --------------------------------------------------------------------------
# Safe output
# --------------------------------------------------------------------------
def partial_path(output: Path) -> Path:
    return output.with_name(output.name + ".partial")


def manifest_path(output: Path) -> Path:
    return output.with_name(output.stem + ".manifest.json")


def check_output_paths(output: Path, force: bool) -> None:
    """Fail before any API call if existing files would be overwritten."""
    if force:
        return
    existing = [p for p in (output, partial_path(output), manifest_path(output)) if p.exists()]
    if existing:
        names = ", ".join(str(p) for p in existing)
        raise AgentRunnerError(
            f"Refusing to overwrite existing output: {names}. These may be paid-for results "
            "or a leftover from an interrupted run. Re-run with --force to replace them "
            "(existing files are kept as timestamped .bak copies)."
        )


def _backup(path: Path) -> None:
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path.rename(path.with_name(f"{path.name}.bak-{stamp}"))


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class AtomicJsonlWriter:
    """Append records to '<output>.partial'; rename to '<output>' only on commit()."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.partial = partial_path(output)
        self._fh: Any = None

    def open(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        _backup(self.partial)  # a stale partial exists only if --force was given
        self._fh = open(self.partial, "x", encoding="utf-8")

    def write(self, record: RunLogRecord) -> None:
        line = json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()

    def commit(self) -> None:
        self.close()
        _backup(self.output)
        os.replace(self.partial, self.output)


def _sdk_version(provider: str) -> str | None:
    try:
        return metadata.version(provider)
    except metadata.PackageNotFoundError:
        return None


def build_manifest(
    *,
    run_id: str,
    started_at: str,
    config: RunnerConfig,
    dataset: RunnerDataset,
    dataset_sha256: str,
    input_path: Path,
    plan: RunPlan,
    summary: RunSummary,
    allow_unreviewed_demo: bool,
) -> dict[str, Any]:
    """Run-level audit record. Contains no secrets."""
    return {
        "run_id": run_id,
        "status": "complete",
        "started_at": started_at,
        "finished_at": _now(),
        "notice": RUN_NOTICE,
        "provider": config.target_provider,
        "model": config.target_model,
        "sdk_version": _sdk_version(config.target_provider),
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "prompt_version": PROMPT_VERSION,
        "input_path": str(input_path),
        "dataset_sha256": dataset_sha256,
        "dataset_version": dataset.dataset_version,
        "generator_model": dataset.generator_model,
        "target_model_equals_generator_model": dataset.generator_model == config.target_model,
        "allow_unreviewed_demo": allow_unreviewed_demo,
        "provisional": plan.provisional,
        "counts": summary.as_dict(),
        "skipped_inputs": [
            {"scenario_id": s.scenario_id, "variant_id": s.variant_id, "reason": s.reason}
            for s in plan.skipped
        ],
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the configured target model against the scenario dataset and log its "
            "decisions and stated confidence. WARNING: this makes paid API calls using "
            "the credentials in .env. No scoring or metrics are computed here."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH,
                        help="Scenario JSON from scenario_generator.py (default: outputs/scenarios.json).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
                        help="JSONL log path (default: outputs/agent_logs.jsonl).")
    parser.add_argument("--force", action="store_true",
                        help="Replace existing output files (they are kept as .bak copies).")
    parser.add_argument("--allow-unreviewed-demo", action="store_true",
                        help="Also run scenarios/variants that are not human-approved. "
                             "Records are marked provisional. For software demos only.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the interactive confirmation of API spend.")
    parser.add_argument("--max-consecutive-failures", type=int, default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
                        help="Abort after this many consecutive failed inputs (default: 5).")
    args = parser.parse_args(argv)
    if args.max_consecutive_failures < 1:
        parser.error("--max-consecutive-failures must be at least 1.")
    return args


def _describe_plan(plan: RunPlan, config: RunnerConfig) -> str:
    max_requests = len(plan.calls) * (config.max_retries + 1)
    lines = [
        f"Planned inputs: {len(plan.calls)} ({sum(c.variant_type == BASE_VARIANT_TYPE for c in plan.calls)} base, "
        f"{sum(c.variant_type != BASE_VARIANT_TYPE for c in plan.calls)} variants); skipped: {len(plan.skipped)}.",
        f"Target: {config.target_provider} / {config.target_model}.",
        f"COST NOTICE: this will make at least {len(plan.calls)} paid API calls "
        f"(up to {max_requests} if retries are needed) with your {config.target_provider} credentials.",
    ]
    if plan.provisional:
        lines.append("PROVISIONAL: some inputs are not human-approved. Results are for software demonstration only.")
    return "\n".join(lines)


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[[RunnerConfig], TargetClient] = build_target_client,
    sleep: Callable[[float], None] = time.sleep,
    input_fn: Callable[[str], str] = input,
) -> int:
    args = parse_args(argv)  # --help exits here, before any config or API activity
    writer: AtomicJsonlWriter | None = None
    try:
        check_output_paths(args.output, args.force)
        dataset, dataset_sha256 = load_dataset(args.input)
        plan = build_run_plan(dataset, args.allow_unreviewed_demo)
        config = load_config()

        print(_describe_plan(plan, config))
        if not args.yes:
            if not sys.stdin.isatty():
                raise AgentRunnerError("Non-interactive session: pass --yes to confirm the API spend.")
            if input_fn("Type 'yes' to proceed: ").strip().lower() != "yes":
                print("Cancelled. No API calls were made.")
                return 1

        client = client_factory(config)
        run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        started_at = _now()
        writer = AtomicJsonlWriter(args.output)
        writer.open()
        summary = run_evaluation(
            plan, client, config, run_id, RetryPolicy(max_retries=config.max_retries),
            write_record=writer.write,
            max_consecutive_failures=args.max_consecutive_failures,
            sleep=sleep,
            progress=print,
        )
        writer.commit()
        mpath = manifest_path(args.output)
        _backup(mpath)
        write_json_atomic(mpath, build_manifest(
            run_id=run_id, started_at=started_at, config=config, dataset=dataset,
            dataset_sha256=dataset_sha256, input_path=args.input, plan=plan,
            summary=summary, allow_unreviewed_demo=args.allow_unreviewed_demo,
        ))
    except RunAborted as exc:
        if writer is not None:
            writer.close()
            print(f"Partial results kept (NOT a completed run): {writer.partial}", file=sys.stderr)
        print(f"Agent run aborted: {exc}", file=sys.stderr)
        return 1
    except AgentRunnerError as exc:
        if writer is not None:
            writer.close()
        print(f"Agent run failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        if writer is not None:
            writer.close()
            print(f"\nInterrupted. Partial results kept (NOT a completed run): {writer.partial}", file=sys.stderr)
        return 130

    print(f"Run {run_id} complete: {summary.success} success, {summary.api_error} api_error, "
          f"{summary.invalid_response} invalid_response of {summary.planned} planned.")
    print(f"Logs: {args.output}\nManifest: {mpath}")
    if plan.provisional:
        print("PROVISIONAL: includes inputs without human approval. Not valid evidence of AI reliability.")
    print("No metrics were computed. Run metrics_engine.py (Step 4) for deterministic scoring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
