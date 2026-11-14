import {
  buildAnswersMessage,
  formatAnswer,
  optionLetter,
  parseOnboarding,
  parseSentAnswers,
  pickOption,
  setCustomText,
  SKIP_MESSAGE,
  type OnboardingStep,
} from "../helpers";

const output = {
  type: "expert_onboarding",
  expert_id: "expert-1",
  greeting: "Hi, I'm Jules.",
  steps: [
    {
      question: "Which outcome should I start with?",
      keyword: "outcome",
      options: ["Pricing", "Churn", " Pricing "],
    },
    {
      question: "Which services do you use?",
      keyword: "services",
      options: ["Gmail", "Slack"],
      allow_multiple: true,
    },
    { question: "  " },
  ],
};

const steps = parseOnboarding(output) as OnboardingStep[];

describe("parseOnboarding", () => {
  it("reads the steps from an object or a JSON string", () => {
    expect(steps).toEqual([
      {
        question: "Which outcome should I start with?",
        keyword: "outcome",
        options: ["Pricing", "Churn"],
        allowMultiple: false,
      },
      {
        question: "Which services do you use?",
        keyword: "services",
        options: ["Gmail", "Slack"],
        allowMultiple: true,
      },
    ]);
    expect(parseOnboarding(JSON.stringify(output))).toEqual(steps);
  });

  it("rejects other outputs and cards without questions", () => {
    expect(parseOnboarding({ type: "error", message: "nope" })).toBeNull();
    expect(parseOnboarding({ type: "expert_onboarding", steps: [] })).toBeNull();
    expect(parseOnboarding("not json")).toBeNull();
  });

  it("falls back to a step keyword", () => {
    const parsed = parseOnboarding({
      type: "expert_onboarding",
      steps: [{ question: "What topic?", options: [] }],
    });
    expect(parsed?.[0].keyword).toBe("step-0");
  });
});

describe("answers", () => {
  it("letters options from A", () => {
    expect([0, 1, 4].map(optionLetter)).toEqual(["A", "B", "E"]);
  });

  it("keeps one pick or the typed text for a single-choice step", () => {
    const picked = pickOption(steps[0], undefined, "Churn");
    expect(picked).toEqual({ picks: ["Churn"], custom: "" });
    expect(setCustomText(steps[0], picked, "Retention")).toEqual({
      picks: [],
      custom: "Retention",
    });
  });

  it("toggles picks in the offered order for a multi-choice step", () => {
    const one = pickOption(steps[1], undefined, "Slack");
    const two = pickOption(steps[1], one, "Gmail");
    expect(two.picks).toEqual(["Gmail", "Slack"]);
    expect(pickOption(steps[1], two, "Slack").picks).toEqual(["Gmail"]);
    expect(formatAnswer({ ...two, custom: "Notion" })).toBe(
      "- Gmail\n- Slack\n- Notion",
    );
  });
});

describe("buildAnswersMessage", () => {
  it("builds the frame the backend reads", () => {
    const message = buildAnswersMessage(steps, {
      outcome: { picks: ["Pricing"], custom: "" },
      services: { picks: ["Gmail", "Slack"], custom: "" },
    });
    expect(message).toBe(
      "**Here are my answers:**\n\n" +
        "> Which outcome should I start with?\n\nPricing\n\n" +
        "> Which services do you use?\n\n- Gmail\n- Slack\n\n" +
        "Please proceed.",
    );
  });
});

describe("parseSentAnswers", () => {
  it("reads the answers back from a reply", () => {
    const message = buildAnswersMessage(steps, {
      outcome: { picks: [], custom: "Retention" },
      services: { picks: ["Gmail", "Slack"], custom: "" },
    });
    expect(parseSentAnswers(message, steps)).toEqual([
      {
        keyword: "outcome",
        question: "Which outcome should I start with?",
        answer: "Retention",
      },
      {
        keyword: "services",
        question: "Which services do you use?",
        answer: "Gmail, Slack",
      },
    ]);
  });

  it("finds nothing in a skip or an unrelated reply", () => {
    expect(parseSentAnswers(SKIP_MESSAGE, steps)).toEqual([]);
    expect(parseSentAnswers("Can you look at churn?", steps)).toEqual([]);
  });
});
