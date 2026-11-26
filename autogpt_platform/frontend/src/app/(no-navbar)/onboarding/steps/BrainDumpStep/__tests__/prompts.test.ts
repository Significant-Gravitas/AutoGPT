import { describe, expect, it } from "vitest";
import {
  BRAIN_DUMP_PROMPT_SETS,
  offYourPlateExampleForRole,
  promptSetNameForRole,
} from "../prompts";

// The ids RoleStep stores as `role`. A set keyed by anything else would never
// be picked, and the user would silently get the Other set.
const ROLE_IDS = [
  "Founder/CEO",
  "Operations",
  "Sales/BD",
  "Marketing",
  "Product/PM",
  "Engineering",
  "HR/People",
  "Other",
];

describe("brain dump prompt sets", () => {
  it.each(ROLE_IDS)("has eight questions and five examples for %s", (role) => {
    const set = BRAIN_DUMP_PROMPT_SETS[promptSetNameForRole(role)];

    expect(set).toBe(
      BRAIN_DUMP_PROMPT_SETS[role as keyof typeof BRAIN_DUMP_PROMPT_SETS],
    );
    expect(set.questions).toHaveLength(8);
    expect(set.examples).toHaveLength(5);
    for (const question of set.questions) {
      expect(question.label).not.toBe("");
      expect(question.text).toMatch(/\?$/);
    }
  });

  it("has a set for every role and nothing else", () => {
    expect(Object.keys(BRAIN_DUMP_PROMPT_SETS).sort()).toEqual(
      [...ROLE_IDS].sort(),
    );
  });

  it.each(["", "Dentist", "Other", "founder/ceo", "toString"])(
    "falls back to the Other set for %j",
    (role) => {
      expect(BRAIN_DUMP_PROMPT_SETS[promptSetNameForRole(role)]).toBe(
        BRAIN_DUMP_PROMPT_SETS.Other,
      );
      expect(promptSetNameForRole(role)).toBe("Other");
    },
  );

  it("splits a label from its question at the first colon only", () => {
    expect(
      BRAIN_DUMP_PROMPT_SETS[promptSetNameForRole("Engineering")].questions[1],
    ).toEqual({
      label: "Around the code",
      text: "Which non-coding chores take your time: updates, docs, tickets, release notes?",
    });
  });

  it("keeps em dashes out of every question and example", () => {
    const copy = Object.values(BRAIN_DUMP_PROMPT_SETS).flatMap((set) => [
      ...set.questions.flatMap((question) => [question.label, question.text]),
      ...set.examples,
      set.offYourPlateExample,
    ]);

    expect(copy.filter((line) => line.includes("—"))).toEqual([]);
  });

  it.each(ROLE_IDS)(
    "has an off-your-plate example for %s, used by the helper",
    (role) => {
      const example = offYourPlateExampleForRole(role);

      expect(example.trim()).not.toBe("");
      expect(example).not.toContain("—");
      expect(example).toBe(
        BRAIN_DUMP_PROMPT_SETS[role as keyof typeof BRAIN_DUMP_PROMPT_SETS]
          .offYourPlateExample,
      );
    },
  );

  it("gives each role its own off-your-plate example", () => {
    const examples = ROLE_IDS.map(offYourPlateExampleForRole);

    expect(new Set(examples).size).toBe(ROLE_IDS.length);
    expect(offYourPlateExampleForRole("Marketing")).toBe(
      "The newsletter. It's always the thing that slips.",
    );
  });

  it.each(["", "Dentist", "founder/ceo", "toString"])(
    "falls back to Other's off-your-plate example for %j",
    (role) => {
      expect(offYourPlateExampleForRole(role)).toBe(
        "The reminders and invoices for my students.",
      );
    },
  );
});
