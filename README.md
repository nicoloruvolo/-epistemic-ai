Epistemic.ai

Evaluating confidence, calibration, and reliability in large language models.

Epistemic.ai is an early-stage research and engineering project exploring whether confidence expressed by large language models (LLMs) provides a reliable signal about the correctness of their answers.

The project aims to develop a reproducible evaluation framework that examines not only whether a model answers correctly, but also whether its expressed confidence is informative, stable, and useful for decision-making.

Project status: Early development. Automated tests pass, but the evaluation methodology, scoring criteria, and empirical findings still require further review and validation.

Research Motivation

An LLM can provide a correct answer with low confidence or an incorrect answer with high confidence. Consequently, answer accuracy alone may be insufficient when a system must decide whether to trust an answer, request verification, or abstain.

Epistemic.ai approaches this problem from a measurement perspective.

Rather than assuming that a model’s stated confidence is meaningful, the project investigates whether that confidence behaves like a useful measurement signal.

Research Questions

The initial research framework focuses on three questions:

1. Validity: Does stated confidence predict answer correctness?
2. Stability: Does confidence remain consistent across prompt variations and repeated evaluation sessions?
3. Decision utility: Can confidence help determine when an answer should be accepted, checked, or rejected?

These questions motivate an evaluation framework connecting model confidence to concepts from psychometrics, statistical measurement, and selective prediction.

Current Architecture

The project currently includes two main Python modules.

scenario_generator.py

Generates synthetic evaluation scenarios using an OpenAI model.

The generated scenarios include draft scoring criteria. These criteria are marked as pending review until a human has reviewed and approved them.

Generated scenarios should not automatically be treated as scientifically validated evaluation items.

agent_runner.py

Runs a target model against evaluation scenarios and records evaluation outputs.

The runner includes safeguards against using scenarios whose scoring criteria have not been approved. An explicit demonstration option can allow unreviewed scenarios for software testing, but those results must be treated as provisional.

Important: Actual API-backed runs can incur charges. Review the configuration and expected usage before executing them.

Planned Evaluation Dimensions

The broader research framework is intended to investigate:

* Calibration: Whether confidence values correspond to observed correctness rates.
* Discrimination: Whether confidence distinguishes correct answers from incorrect ones.
* Stability: Whether confidence measurements remain consistent across repeated sessions.
* Prompt invariance: Whether confidence changes substantially when prompt wording changes.
* Decision utility: Whether confidence supports useful acceptance, verification, or abstention policies.

These are research objectives, not claims that every dimension has already been implemented or validated.

Technology Stack

* Python
* OpenAI API
* Anthropic API integration dependencies
* Pydantic
* pytest
* Statistical analysis and visualization tools planned for subsequent research stages

Getting Started

1. Clone the repository

git clone https://github.com/nicoloruvolo/-epistemic-ai.git
cd -epistemic-ai

2. Create a virtual environment

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

python -m pip install -r requirements-dev.txt

4. Configure environment variables

Copy the example configuration:

cp .env.example .env

Edit .env and add your own API credentials if you intend to run API-backed experiments.

Never commit .env, publish API keys, or include secrets in source code.

The automated tests are designed to exercise simulated behavior and should not require paid API calls.

5. Run the test suite

python -m pytest tests/ -v

The initial development checkpoint passed 115 automated tests in the original development environment. Results may vary as the code and dependencies evolve.

Cost and Safety Considerations

API-backed scenario generation and model evaluation may incur usage charges.

Before running experiments:

1. Confirm which model is configured.
2. Review the number of scenarios and planned API calls.
3. Check the provider’s current pricing and account billing settings.
4. Keep credentials in environment variables.
5. Avoid sending sensitive personal or confidential information to external models.

No API call is necessary to run the local automated test suite.

Scientific Limitations

The current project is an experimental framework, not a validated psychometric instrument.

Important limitations include:

* Generated scenarios and scoring criteria require human review.
* Synthetic or model-generated evaluation items may contain ambiguities, biases, or errors.
* Results from a single model, prompt, or domain cannot establish general validity.
* Confidence stability does not, by itself, establish measurement validity.
* Statistical metrics require careful interpretation and appropriate uncertainty estimates.
* Empirical results must be collected and independently examined before substantive scientific conclusions can be drawn.

The methodology will need to be refined through theoretical justification, pilot testing, appropriate statistical analyses, and documented validation procedures.

Roadmap

* Establish the initial Python project structure.
* Implement initial scenario-generation and evaluation modules.
* Add automated tests and basic safeguards.
* Initialize local Git version control.
* Publish the initial repository to GitHub.
* Document the installation and evaluation workflow.
* Review and approve scenario scoring criteria.
* Refine the experimental design and measurement definitions.
* Implement and verify the planned statistical analyses.
* Conduct controlled pilot evaluations.
* Document results, limitations, and reproducibility procedures.

Contributing

The project is in an early research and development stage.

Contributions involving evaluation methodology, psychometrics, statistical modeling, machine learning, and reproducible experimentation may be particularly valuable.

Before interpreting evaluation outputs as scientific evidence, contributors should review the scoring criteria, assumptions, and limitations of the experimental design.

License

No license has been specified yet. Until a license is added, do not assume that the repository is available for unrestricted reuse or redistribution.