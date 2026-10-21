# Epistemic.ai

**Evaluating confidence, calibration, and reliability in large language models.**

Epistemic.ai is an early-stage research and engineering project exploring whether confidence expressed by large language models (LLMs) provides a reliable signal about the correctness of their answers.

The goal is to develop a reproducible evaluation framework that examines whether model confidence is informative, stable, and useful for decision-making.

> **Project status:** Early development. Automated tests pass, but the evaluation methodology, scoring criteria, and empirical findings still require further review and validation.

## Research Motivation

An LLM can provide a correct answer with low confidence or an incorrect answer with high confidence. Answer accuracy alone may therefore be insufficient when deciding whether to trust an answer, request verification, or abstain.

Epistemic.ai approaches this problem from a measurement perspective. Rather than assuming that a model's stated confidence is meaningful, the project investigates whether confidence behaves like a useful measurement signal.

## Research Questions

The initial research framework focuses on three questions:

1. **Validity:** Does stated confidence predict answer correctness?
2. **Stability:** Does confidence remain consistent across prompt variations and repeated evaluation sessions?
3. **Decision utility:** Can confidence help determine when an answer should be accepted, checked, or rejected?

These questions connect model evaluation with psychometrics, statistical measurement, and selective prediction.

## Current Architecture

### `scenario_generator.py`

Generates synthetic evaluation scenarios using an OpenAI model.

The generated scenarios include draft scoring criteria that remain pending human review. Generated scenarios should not automatically be treated as scientifically validated evaluation items.

### `agent_runner.py`

Runs a target model against evaluation scenarios and records evaluation outputs.

The runner includes safeguards against using scenarios whose scoring criteria have not been approved. An explicit demonstration option can permit unreviewed scenarios for software testing, but the resulting records must be treated as provisional.

**Important:** Actual API-backed runs can incur charges. Review the configuration and expected usage before executing them.

## Planned Evaluation Dimensions

The broader research framework aims to investigate:

- **Calibration:** Whether confidence corresponds to observed correctness rates.
- **Discrimination:** Whether confidence distinguishes correct answers from incorrect ones.
- **Stability:** Whether confidence measurements remain consistent across repeated sessions.
- **Prompt invariance:** Whether confidence changes when prompt wording changes.
- **Decision utility:** Whether confidence supports useful acceptance, verification, or abstention policies.

These are research objectives, not claims that every dimension has already been implemented or validated.

## Technology Stack

- Python
- OpenAI API
- Anthropic API integration dependencies
- Pydantic
- pytest
- Statistical analysis and visualization tools planned for subsequent research stages

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/nicoloruvolo/-epistemic-ai.git
cd ./-epistemic-ai
