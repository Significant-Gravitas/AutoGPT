from backend.blocks.agent_evaluator_helpers import (
    MAX_SCORE,
    MIN_SCORE,
    EvaluationCriterion,
    build_user_prompt,
    clamp_score,
    default_criteria,
    evaluation_response_format,
    normalize_criteria_scores,
    validate_criteria_names,
    weighted_overall,
)


def test_default_criteria_base():
    criteria = default_criteria(has_reference=False, has_design=False)
    names = [c.name for c in criteria]
    assert names == ["Goal Achievement", "Completeness", "Relevance", "Coherence"]


def test_default_criteria_adds_reference_and_design():
    criteria = default_criteria(has_reference=True, has_design=True)
    names = {c.name for c in criteria}
    assert "Accuracy vs Reference" in names
    assert "Design Quality" in names
    reference = next(c for c in criteria if c.name == "Accuracy vs Reference")
    assert reference.weight == 2.0


def test_clamp_score_valid_and_bounds():
    assert clamp_score(50) == 50.0
    assert clamp_score(-10) == MIN_SCORE
    assert clamp_score(150) == MAX_SCORE


def test_clamp_score_invalid_returns_min():
    assert clamp_score(None) == MIN_SCORE
    assert clamp_score("not-a-number") == MIN_SCORE


def test_weighted_overall_uses_weights():
    criteria = [
        EvaluationCriterion(name="A", weight=1.0),
        EvaluationCriterion(name="B", weight=3.0),
    ]
    evaluations = [
        {"name": "A", "score": 100},
        {"name": "B", "score": 60},
    ]
    # (100*1 + 60*3) / 4 = 70
    assert weighted_overall(evaluations, criteria) == 70.0


def test_weighted_overall_empty_returns_min():
    assert weighted_overall([], []) == MIN_SCORE


def test_validate_criteria_names_match():
    criteria = [EvaluationCriterion(name="Goal Achievement")]
    evaluations = [{"name": "goal achievement", "score": 90}]
    assert validate_criteria_names(evaluations, criteria) is None


def test_validate_criteria_names_missing():
    criteria = [
        EvaluationCriterion(name="Goal Achievement"),
        EvaluationCriterion(name="Completeness"),
    ]
    evaluations = [{"name": "Goal Achievement", "score": 90}]
    error = validate_criteria_names(evaluations, criteria)
    assert error is not None
    assert "missing criteria" in error


def test_validate_criteria_names_unexpected():
    criteria = [EvaluationCriterion(name="Goal Achievement")]
    evaluations = [
        {"name": "Goal Achievement", "score": 90},
        {"name": "Hallucinated Criterion", "score": 50},
    ]
    error = validate_criteria_names(evaluations, criteria)
    assert error is not None
    assert "unexpected criteria" in error


def test_normalize_criteria_scores():
    criteria = [EvaluationCriterion(name="Goal Achievement", weight=2.0)]
    evaluations = [
        {"name": "Goal Achievement", "score": 200, "reasoning": "great"},
    ]
    normalized = normalize_criteria_scores(evaluations, criteria)
    assert normalized == [
        {
            "name": "Goal Achievement",
            "score": MAX_SCORE,
            "weight": 2.0,
            "reasoning": "great",
        }
    ]


def test_normalize_criteria_scores_unknown_name_default_weight():
    normalized = normalize_criteria_scores([{"name": "Unknown", "score": 10}], [])
    assert normalized[0]["weight"] == 1.0
    assert normalized[0]["reasoning"] == ""


def test_build_user_prompt_includes_optional_sections():
    criteria = [EvaluationCriterion(name="Goal Achievement", description="desc")]
    prompt = build_user_prompt(
        goal="do X",
        agent_output="did X",
        expected_output="X reference",
        agent_design="block graph",
        criteria=criteria,
    )
    assert "## GOAL / TASK" in prompt
    assert "## EXPECTED / REFERENCE ANSWER" in prompt
    assert "## AGENT DESIGN" in prompt
    assert "Goal Achievement" in prompt


def test_build_user_prompt_omits_absent_sections():
    prompt = build_user_prompt(
        goal="",
        agent_output="out",
        expected_output="",
        agent_design="",
        criteria=[EvaluationCriterion(name="Relevance")],
    )
    assert "(not provided)" in prompt
    assert "## EXPECTED / REFERENCE ANSWER" not in prompt
    assert "## AGENT DESIGN" not in prompt


def test_evaluation_response_format_keys():
    fmt = evaluation_response_format()
    assert set(fmt.keys()) == {
        "criteria_evaluations",
        "strengths",
        "weaknesses",
        "suggestions",
        "summary",
    }
