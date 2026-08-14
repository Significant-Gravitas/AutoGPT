import { describe, expect, test } from "vitest";
import {
  buildTranscript,
  EMPTY_DRAFT,
  normalizeDraftForNaming,
  previousStep,
  RAISE_PROMPTS,
  VOICE_SKIPPED_LABEL,
  type RaiseDraft,
} from "./helpers";

const namingDraft: RaiseDraft = {
  ...EMPTY_DRAFT,
  name: "Otto",
  voiceLabel: "Concise and direct",
  step: "review",
};

describe("normalizeDraftForNaming", () => {
  test("drops a leftover first job from an abandoned regular draft", () => {
    const normalized = normalizeDraftForNaming({
      ...EMPTY_DRAFT,
      name: "Otto",
      firstJob: { id: "listing-1", name: "SEO Blog Writer" },
      firstJobSkipped: false,
      step: "review",
    });

    expect(normalized.firstJob).toBeNull();
    expect(normalized.firstJobSkipped).toBe(false);
    expect(normalized.step).toBe("review");
  });

  test("rewrites the firstJob step to review since naming has no job step", () => {
    const normalized = normalizeDraftForNaming({
      ...EMPTY_DRAFT,
      name: "Otto",
      step: "firstJob",
    });

    expect(normalized.step).toBe("review");
  });

  test("leaves earlier steps where they are", () => {
    expect(normalizeDraftForNaming({ ...EMPTY_DRAFT, step: "name" }).step).toBe(
      "name",
    );
    expect(
      normalizeDraftForNaming({ ...EMPTY_DRAFT, step: "voice" }).step,
    ).toBe("voice");
  });
});

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
    const texts = buildTranscript({
      ...namingDraft,
      firstJobSkipped: true,
    }).map((message) => message.text);

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
