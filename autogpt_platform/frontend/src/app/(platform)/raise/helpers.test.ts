import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";
import { beforeEach, describe, expect, test } from "vitest";
import {
  assembledKit,
  EMPTY_DRAFT,
  getExpertLimitCode,
  kitBudgetLabel,
  kitToolsLabel,
  loadDraft,
  raisedIdentity,
  type RaiseDraft,
  resolveVoicePreferences,
  saveDraft,
  VOICE_SKIPPED_LABEL,
  voiceSummaryLabel,
} from "./helpers";

const samples: VoiceSample[] = [
  { label: "Direct", text: "Do this next." },
  { label: "Warm", text: "Let's work through this together." },
];

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

  test("formats weekly budget and tool labels for the soul preview", () => {
    expect(kitBudgetLabel(null)).toBeNull();
    expect(kitBudgetLabel({ weeklyBudget: null, attachments: [] })).toBeNull();
    expect(kitBudgetLabel({ weeklyBudget: 0, attachments: [] })).toBe(
      "No weekly limit",
    );
    expect(kitBudgetLabel({ weeklyBudget: 500, attachments: [] })).toBe(
      "500 credits ($5/week)",
    );
    expect(
      kitToolsLabel({
        weeklyBudget: null,
        attachments: [
          {
            kind: "skill",
            source: "library",
            id: "seo-audit",
            name: "SEO audit",
          },
        ],
      }),
    ).toBe("SEO audit");
  });

  test("assembles preview kit from answered budget and attachments", () => {
    expect(assembledKit(EMPTY_DRAFT)).toBeNull();
    expect(
      assembledKit({
        ...EMPTY_DRAFT,
        budget: { credits: 500 },
        marketplace: [
          {
            kind: "workflow",
            source: "marketplace",
            id: "listing-1",
            name: "SEO Blog Writer",
          },
        ],
      }),
    ).toEqual({
      weeklyBudget: 500,
      attachments: [
        {
          kind: "workflow",
          source: "marketplace",
          id: "listing-1",
          name: "SEO Blog Writer",
        },
      ],
    });
  });

  test("keeps a skipped budget distinct from an unanswered one", () => {
    expect(assembledKit({ ...EMPTY_DRAFT, budget: { credits: null } })).toEqual(
      { weeklyBudget: null, attachments: [] },
    );
  });
});

describe("restoring a persisted draft", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  test("backfills the skipped-voice label when the draft is past the voice beat", () => {
    saveDraft({ ...EMPTY_DRAFT, step: "budget", voiceLabel: null });

    expect(loadDraft().voiceLabel).toBe(VOICE_SKIPPED_LABEL);
  });

  test("leaves the voice beat unanswered when the draft has not reached it", () => {
    saveDraft({ ...EMPTY_DRAFT, step: "about", voiceLabel: null });

    expect(loadDraft().voiceLabel).toBeNull();
  });

  test("keeps an explicitly picked voice label", () => {
    saveDraft({ ...EMPTY_DRAFT, step: "budget", voiceLabel: "Direct" });

    expect(loadDraft().voiceLabel).toBe("Direct");
  });

  test("moves a draft parked on the retired kit step onto budget", () => {
    saveStepFromEarlierBuild("kit");

    expect(loadDraft().step).toBe("budget");
  });

  test("restarts the flow when the stored step is not a known step", () => {
    saveStepFromEarlierBuild("space-invaders");

    expect(loadDraft().step).toBe(EMPTY_DRAFT.step);
  });
});

function saveStepFromEarlierBuild(step: string) {
  saveDraft({ ...EMPTY_DRAFT, step } as RaiseDraft);
}
