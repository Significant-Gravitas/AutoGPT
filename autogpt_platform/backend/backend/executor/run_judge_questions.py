"""Question set for the TypeSafe Jev run judge.

Kept separate from ``run_judge.py`` so the rubric text can be reviewed and
edited on its own. Wording is neutral and evidence-referencing: every option
names the JSON fields in the Jev ``state`` it should be judged from.
"""

from typesafe_sdk import Choice, Noul, Score

# Question keys are part of the persisted record shape: keep them stable.
DELIVERED = "delivered"
ERRORS_VS_OUTCOME = "errors_vs_outcome"
FAILURE_CAUSE = "failure_cause"
USER_ACTION_NEEDED = "user_action_needed"
EXTERNAL_SIDE_EFFECTS = "external_side_effects"
OUTPUT_QUALITY = "output_quality"

QUESTION_KEYS = (
    DELIVERED,
    ERRORS_VS_OUTCOME,
    FAILURE_CAUSE,
    USER_ACTION_NEEDED,
    EXTERNAL_SIDE_EFFECTS,
    OUTPUT_QUALITY,
)

_EVIDENCE_PREAMBLE = (
    "The state is a JSON record of one run of an automation graph. "
    "graph_info.name and graph_info.description say what the graph promises the "
    "user. nodes[] lists each step with block_name, block_description, "
    "execution_count, error_count, recent_errors[].error, recent_inputs[] and "
    "recent_outputs[].output_data. Steps with is_graph_output=true are the "
    "graph's final outputs and carry longer, less truncated output_data; other "
    "steps' data is truncated to 100 characters. input_output_data holds the "
    "raw inputs/outputs of AgentInputBlock/AgentOutputBlock steps. "
    "overall_status has graph_execution_status, graph_error, total_errors, "
    "total_executions and execution_time_seconds. Judge only from this evidence."
)

DELIVERED_OPTIONS: dict[str, str] = {
    "delivered": (
        "The evidence shows the outcome promised by graph_info.description was "
        "produced: the is_graph_output steps and input_output_data contain the "
        "kind of result the description promises, and overall_status shows no "
        "graph_error that would invalidate it."
    ),
    "partially_delivered": (
        "Some, but not all, of the promised outcome is present in the evidence: "
        "e.g. only part of the expected results appear, some outputs are empty "
        "or degraded, or a stated sub-goal has no corresponding output."
    ),
    "not_delivered": (
        "The evidence shows the promised outcome was not produced: graph_error is "
        "set, the is_graph_output steps have no meaningful output_data, or the "
        "outputs contradict the description."
    ),
    "cannot_tell_from_evidence": (
        "The evidence does not show the outputs (no is_graph_output output_data, "
        "outputs truncated beyond interpretation, or a description too vague to "
        "compare against), so delivery cannot be judged either way."
    ),
}

ERRORS_VS_OUTCOME_OPTIONS: dict[str, str] = {
    "no_errors": (
        "overall_status.total_errors is 0, graph_error is null and no node has "
        "recent_errors."
    ),
    "errors_recovered_outcome_unaffected": (
        "One or more nodes have recent_errors or error_count > 0, but later "
        "executions of those nodes or other nodes produced the outputs, so the "
        "final outputs are unaffected."
    ),
    "errors_degraded_outcome": (
        "Errors occurred and the final outputs exist but are incomplete or lower "
        "quality because of them."
    ),
    "errors_caused_failure": (
        "Errors (node errors or graph_error) are the reason the promised outcome "
        "was not produced."
    ),
}

FAILURE_CAUSE_OPTIONS: dict[str, str] = {
    "not_applicable": (
        "Choose this when the run delivered its outcome, or when no failure is "
        "visible in the evidence."
    ),
    "bad_user_input": (
        "The recent_inputs / input_output_data show the user supplied input that "
        "is empty, malformed, nonsensical, or outside what the graph accepts, and "
        "the errors or empty outputs follow from that."
    ),
    "missing_credential_or_integration": (
        "recent_errors mention missing, invalid, expired or unauthorized "
        "credentials, API keys, tokens, or an integration that is not connected."
    ),
    "external_service_failure": (
        "recent_errors show a third-party service or network failing (timeouts, "
        "5xx, rate limits, unavailable endpoints) despite valid credentials and "
        "input."
    ),
    "agent_design_or_wiring": (
        "The graph itself is at fault: node_relations wire outputs to the wrong "
        "inputs, required inputs are never supplied, a step's block cannot do what "
        "the description needs, or steps run in an order that cannot succeed."
    ),
    "platform_bug": (
        "recent_errors or graph_error show an internal error of the platform "
        "itself (unexpected exception, traceback, missing block, internal server "
        "error) that neither the user's input nor the graph design explains."
    ),
}

USER_ACTION_NEEDED_OPTIONS: dict[str, str] = {
    "none": (
        "No user action is indicated: the outcome was delivered, or the failure "
        "is transient and outside the user's control."
    ),
    "fix_the_input": (
        "The user should change the input they provided (see recent_inputs and "
        "input_output_data) and run again."
    ),
    "connect_an_integration": (
        "The user should add, reconnect or re-authorize a credential or "
        "integration named in recent_errors."
    ),
    "add_credits": (
        "recent_errors or graph_error say the run stopped for lack of credits, "
        "balance, quota or an unavailable plan feature."
    ),
    "rebuild_the_agent": (
        "The graph's steps or wiring need to change before a re-run can succeed."
    ),
}

EXTERNAL_SIDE_EFFECTS_OPTIONS: dict[str, str] = {
    "none": (
        "No step's block_name/block_description/outputs indicate it sent, posted, "
        "wrote, deleted or purchased anything outside the platform; or such steps "
        "never produced output."
    ),
    "intended_only": (
        "Steps that act on external systems (send, post, publish, create, update, "
        "delete, pay) produced outputs consistent with a single intended action "
        "matching graph_info.description."
    ),
    "unintended_or_repeated": (
        "Outputs or execution_count of an externally-acting step indicate the "
        "action ran more times than the description implies, acted on the wrong "
        "target, or happened despite the run failing."
    ),
}

# Ordered lowest-first; index == Jev score level.
OUTPUT_QUALITY_LEVELS: list[str] = [
    "unusable: the visible graph outputs are empty, error text, placeholders, or "
    "unrelated to graph_info.description; nothing here is usable by the user.",
    "poor: outputs exist and relate to the task but are largely wrong, "
    "incomplete, or malformed; the user would have to redo most of the work.",
    "acceptable: outputs do the core of what the description promises with "
    "noticeable gaps, errors, or rough formatting the user would need to fix.",
    "good: outputs fulfil the description with only minor flaws that would not "
    "stop the user from using them as-is.",
    "excellent: outputs fully and cleanly fulfil the description, are complete, "
    "well-formed and directly usable with no visible flaws.",
]

_OUTPUT_QUALITY_INSTRUCTIONS = (
    "Rate the quality of the graph's visible final outputs: the output_data of "
    "steps with is_graph_output=true and the *_outputs entries of "
    "input_output_data, measured against graph_info.description. Judge only "
    "what is visible. If no final outputs are visible, choose the lowest level "
    "you can honestly support (usually the lowest) with low confidence; the "
    "delivered question's cannot_tell_from_evidence option is the place to "
    "record that the outputs were not visible."
)


def build_questions() -> dict[str, Choice | Score | Noul]:
    """The fixed question set sent to Jev, keyed as persisted in stats.judge."""
    return {
        DELIVERED: Choice(
            instructions=(
                f"{_EVIDENCE_PREAMBLE} Did this run deliver what "
                "graph_info.description promises the user? Compare the "
                "is_graph_output steps' output_data and input_output_data with "
                "the description; overall_status.graph_execution_status being "
                "COMPLETED is not by itself evidence of delivery."
            ),
            criteria=DELIVERED_OPTIONS,
        ),
        ERRORS_VS_OUTCOME: Choice(
            instructions=(
                "Using nodes[].error_count, nodes[].recent_errors, "
                "overall_status.total_errors and overall_status.graph_error, "
                "how did errors during this run relate to its final outcome?"
            ),
            criteria=ERRORS_VS_OUTCOME_OPTIONS,
        ),
        FAILURE_CAUSE: Choice(
            instructions=(
                "If the run did not deliver its promised outcome, what does the "
                "evidence (recent_errors[].error, graph_error, recent_inputs, "
                "node_relations) show as the primary cause? If the run "
                "delivered, answer not_applicable."
            ),
            criteria=FAILURE_CAUSE_OPTIONS,
        ),
        USER_ACTION_NEEDED: Choice(
            instructions=(
                "Based on the errors, inputs and outputs in the evidence, what "
                "single action, if any, would the user need to take before "
                "running this graph again?"
            ),
            criteria=USER_ACTION_NEEDED_OPTIONS,
        ),
        EXTERNAL_SIDE_EFFECTS: Choice(
            instructions=(
                "Looking at each step's block_name, block_description, "
                "execution_count and recent_outputs, what effects on external "
                "systems (messages sent, content posted, records written, "
                "payments) does the evidence show this run had?"
            ),
            criteria=EXTERNAL_SIDE_EFFECTS_OPTIONS,
        ),
        OUTPUT_QUALITY: Score(
            instructions=_OUTPUT_QUALITY_INSTRUCTIONS,
            criteria=OUTPUT_QUALITY_LEVELS,
        ),
    }
