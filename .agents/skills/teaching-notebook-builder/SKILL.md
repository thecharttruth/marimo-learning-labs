---
name: teaching-notebook-builder
description: Create, edit, or audit notebooks that should teach a subject clearly, including marimo, Jupyter, Colab, MoLab, QuantConnect, research, data science, coding, science, finance, trading, or other educational notebooks. Use when the user asks for a notebook to be instructional, beginner-friendly, high-school-friendly, explanatory, tutorial-like, curriculum-like, or to teach the topic rather than merely run code.
---

# Teaching Notebook Builder

## Goal

Make notebooks teach the subject, not just execute code. Keep the tone clear enough for a bright high-school student while preserving real rigor.

Default audience: curious beginner unless the user specifies otherwise.

## Workflow

1. Identify the subject, audience, and learning outcome.
2. Inspect the existing notebook before editing.
3. Build a teaching spine:
   - what problem the notebook solves
   - why the problem matters
   - what the reader will learn
   - what inputs become outputs
   - how to judge whether results are trustworthy
4. Put a plain-English map near the top.
5. Define key words before using them heavily.
6. Put short teaching markdown before complex code or controls.
7. Make every important table/chart/output answer:
   - what this shows
   - how to read it
   - what good, bad, or suspicious results look like
8. Add student exercises or next experiments at the end.
9. Verify the notebook and update project docs if the workflow changed.

## Teaching Style

Use plain language without becoming childish.

Prefer:

- "features are clues"
- "target is the answer the model learns to predict"
- "backtest is a historical replay"
- "weights are the percent of the portfolio assigned to each asset"
- "validation is a fair test on data the model did not train on"

Avoid unexplained jargon. If jargon is useful, define it once, then use it consistently.

Keep paragraphs short. Use bullets for controls, metrics, caveats, and learning steps. Use formulas only after the plain-English explanation, and label them as optional math notes.

## Notebook Structure

For most teaching notebooks, prefer this order:

1. Title and big question
2. Plain-English map
3. Mini glossary
4. Configuration and controls
5. Data loading and provenance
6. Feature or input construction
7. Method/model/algorithm explanation
8. Validation or evaluation design
9. Results and diagnostics
10. Visual interpretation
11. Limitations and safety caveats
12. Student exercises and next experiments

Adjust the section names to the domain. Do not force trading language into non-trading notebooks.

## Domain Transfer

Teach the actual domain, not only the tool.

- For machine learning notebooks, explain features, targets, leakage, validation, baselines, diagnostics, and overfitting.
- For trading or finance notebooks, explain lookahead bias, execution timing, transaction costs, benchmarks, drawdowns, and out-of-sample gates.
- For science notebooks, explain units, assumptions, measurement limits, uncertainty, and what would falsify the conclusion.
- For coding notebooks, explain the mental model, API shape, failure modes, tests, and debugging path.
- For data analysis notebooks, explain provenance, missing data, transformations, chart interpretation, and caveats.

When the domain is high-stakes, include a clear educational-use caveat and avoid presenting notebook output as advice.

## Controls And Outputs

For each control, explain:

- what it changes
- why someone would change it
- what a safe beginner default is
- what can go wrong when it is pushed too far

For each metric, explain:

- what it measures
- whether higher or lower is better
- what comparison or benchmark matters

For each chart, add a title/subtitle or nearby note that tells the reader what to inspect first.

## Validation Integrity

If the notebook evaluates a model, strategy, or experiment, teach the validation design explicitly.

Avoid random splits for time-series or causal workflows. Use time-aware validation when time matters. In trading, forecasting, and similar workflows, document lookahead prevention, purging/embargoes if used, execution timing, and final holdout rules.

Include simple checks that make mistakes visible: baseline comparison, leakage warnings, sample-window dates, number of test periods, and data provenance.

## Marimo And Cloud Notes

For live marimo sessions, use the marimo-pair workflow and edit through the live runtime, not by directly editing the saved `.py` file behind the kernel.

For cloud notebooks, do not call the work finished until the cloud notebook is saved, pulled/exported back to the durable local project folder, compared, and scanned for secrets. If cloud sync cannot be verified, report it as a remaining gate.

## Verification

Before calling a teaching notebook ready:

- run the notebook checker or syntax check available for the notebook format
- scan for secrets
- scan for stale placeholders or unexplained jargon
- check that the first screen teaches the big question and map
- check that final outputs explain how to interpret results
- update `README.md`, `PLAN.md`, or metadata when the teaching workflow changed
