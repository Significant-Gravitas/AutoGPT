import type { Expert } from "@/app/api/__generated__/models/expert";
import type { RaiseResult } from "@/app/api/__generated__/models/raiseResult";
import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";
import { describe, expect, test } from "vitest";
import {
  buildTranscript,
  EMPTY_DRAFT,
  getExpertLimitCode,
  getFirstJobFailureToast,
  getRaiseErrorToast,
  previousStep,
  RAISE_PROMPTS,
  raisedIdentity,
  resolveVoicePreferences,
  VOICE_SKIPPED_LABEL,
  voiceSummaryLabel,
  type RaiseDraft,
} from "./helpers";

const samples: VoiceSample[] = [
  { label: "Direct", text: "Do this next." },
  { label: "Warm", text: "Let's work through this together." },
];

const raisedExpert: Expert = {
  id: "raised-1",
  name: "Otto",
  avatar_url: null,
  role: "",
  tagline: null,
  bio: null,
  skills: [],
  identity: "I'm Otto, raised by you. I learn how you work and grow with you.",
  voice_preferences: "",
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [],
};

describe("raise helpers", () => {
  test("labels preset and custom voice choices", () => {
    expect(voiceSummaryLabel({ choice: "a" }, samples)).toBe("Direct");
    expect(voiceSummaryLabel({ choice: "b" }, samples)).toBe("Warm");
    expect(
      voiceSummaryLabel(
        { choice: "custom", customText: "My own sample" },
        samples,
      ),
    ).toBe("My own writing sample");
  });

  test("resolves a preset and rejects a blank custom voice", () => {
    expect(resolveVoicePreferences({ choice: "a" }, samples)).toContain(
      "Preferred writing style: Direct.",
    );
    expect(
      resolveVoicePreferences({ choice: "custom", customText: "   " }, samples),
    ).toBeNull();
  });

  test("builds the same complete raised identity shown by the backend", () => {
    expect(raisedIdentity("Otto")).toBe(
      "I'm Otto, raised by you. I learn how you work and grow with you.",
    );
  });

  test("extracts a structured expert-limit code safely", () => {
    expect(
      getExpertLimitCode({
        detail: { code: "raised_expert_lifetime_limit", limit: 100 },
      }),
    ).toBe("raised_expert_lifetime_limit");
    expect(getExpertLimitCode({ detail: "legacy error" })).toBeNull();
    expect(getExpertLimitCode(null)).toBeNull();
  });

  test("describes an unavailable first job as partial success", () => {
    const result: RaiseResult = {
      expert: raisedExpert,
      first_job_installed: false,
      first_job_failure_reason: "unavailable",
    };

    expect(
      getFirstJobFailureToast(result, {
        id: "listing-1",
        name: "SEO Blog Writer",
      }),
    ).toMatchObject({
      title: "SEO Blog Writer is no longer available",
      description:
        "Otto is ready. You can choose another first job from their page.",
    });
  });

  test("maps lifetime, active-team, and generic submission errors", () => {
    expect(
      getRaiseErrorToast(
        {
          status: 409,
          response: { detail: { code: "raised_expert_lifetime_limit" } },
        },
        "Otto",
      ).title,
    ).toBe("Expert creation limit reached");
    expect(
      getRaiseErrorToast(
        {
          status: 409,
          response: { detail: { code: "active_expert_limit" } },
        },
        "Otto",
      ).title,
    ).toBe("Your team is full");
    expect(getRaiseErrorToast(new Error("offline"), "Otto").title).toBe(
      "Couldn't raise Otto",
    );
  });
});

const namingDraft: RaiseDraft = {
  ...EMPTY_DRAFT,
  name: "Otto",
  voiceLabel: "Concise and direct",
  step: "review",
};

describe("buildTranscript in naming mode", () => {
  test("opens with the naming opener instead of the introduction", () => {
    const messages = buildTranscript(EMPTY_DRAFT, true);
    expect(messages[0].text).toBe(RAISE_PROMPTS.namingOpener);
  });

  test("goes from voice straight to review with no first-job prompt", () => {
    const texts = buildTranscript(namingDraft, true).map(
      (message) => message.text,
    );

    expect(texts).toEqual([
      RAISE_PROMPTS.namingOpener,
      "Otto",
      RAISE_PROMPTS.voice("Otto"),
      "Concise and direct",
      RAISE_PROMPTS.review,
    ]);
    expect(texts).not.toContain(RAISE_PROMPTS.firstJob);
  });

  test("labels a skipped voice the same way the regular flow does", () => {
    const texts = buildTranscript(
      { ...namingDraft, voiceLabel: null },
      true,
    ).map((message) => message.text);

    expect(texts).toContain(VOICE_SKIPPED_LABEL);
  });

  test("the regular flow still includes the first-job block at review", () => {
    const texts = buildTranscript(namingDraft).map((message) => message.text);

    expect(texts).toContain(RAISE_PROMPTS.firstJob);
  });
});

describe("previousStep in naming mode", () => {
  test("review goes back to voice, skipping firstJob", () => {
    expect(previousStep("review", true)).toBe("voice");
  });

  test("voice goes back to name, and name stays put", () => {
    expect(previousStep("voice", true)).toBe("name");
    expect(previousStep("name", true)).toBe("name");
  });

  test("the regular flow still walks review back to firstJob", () => {
    expect(previousStep("review")).toBe("firstJob");
  });
});
