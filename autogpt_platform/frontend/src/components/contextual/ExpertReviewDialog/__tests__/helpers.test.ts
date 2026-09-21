import type { ExpertPackagePreview } from "@/app/api/__generated__/models/expertPackagePreview";
import { describe, expect, test } from "vitest";
import {
  buildDraftFromPreview,
  buildEditsFromDraft,
  getBlockingReason,
  getWorkflowSourceLabel,
  serializeExpertEdits,
  type ExpertReviewDraft,
} from "../helpers";

// The dialog's own test covers the wiring; these are the edges it cannot
// reach — a preview that never arrived, and what the form field really holds.
const preview: ExpertPackagePreview = {
  manifest: { identity: { name: "Maria" } },
  avatar_kind: "none",
  workflows: [
    { index: 0, name: "Calendar", source: "store", schedule_cron: "0 9 * * *" },
    { index: 1, name: "Poster", source: "graph" },
  ],
};

test("a draft survives a preview that has not arrived", () => {
  expect(buildDraftFromPreview(null)).toEqual({
    name: "",
    removedSkillSlugs: [],
    removedWorkflowIndices: [],
    scheduledIndices: [],
  });
});

test("the name is trimmed and only kept crons are named", () => {
  const draft = {
    ...buildDraftFromPreview(preview),
    name: "  Ada  ",
    removedWorkflowIndices: [0],
  };

  expect(buildEditsFromDraft(draft, preview)).toEqual({
    name: "Ada",
    removed_skill_slugs: [],
    removed_workflow_indices: [0],
    workflows: [],
  });
});

test("the edits travel as the JSON the form field takes", () => {
  const edits = buildEditsFromDraft(buildDraftFromPreview(preview), preview);

  expect(JSON.parse(serializeExpertEdits(edits))).toEqual(edits);
  expect(edits.workflows).toEqual([{ index: 0, schedule_enabled: true }]);
});

describe("getBlockingReason", () => {
  const whole = buildDraftFromPreview(preview);
  const cases: [ExpertReviewDraft, ExpertPackagePreview, string | null][] = [
    [whole, preview, null],
    [{ ...whole, name: "  " }, preview, "Give this expert a name."],
    [
      whole,
      { ...preview, errors: [{ code: "cap", message: "Full." }] },
      "Fix the problems above to continue.",
    ],
  ];

  test.each(cases)("case %#", (draft, current, expected) => {
    expect(getBlockingReason(draft, current)).toBe(expected);
  });
});

// A workflow without a stored listing is what the file carries on import, but
// on publish it is the admin's own agent, and only the route knows whether it
// has been published since. Neither reading gets to call it unpublished.
describe("getWorkflowSourceLabel", () => {
  const cases: [
    "import" | "publish",
    "store" | "graph" | "unresolvable",
    string,
  ][] = [
    ["import", "store", "Marketplace"],
    ["import", "graph", "From file"],
    ["import", "unresolvable", "Can't import"],
    ["publish", "store", "Marketplace"],
    ["publish", "graph", "Your agent"],
    ["publish", "unresolvable", "Can't publish"],
  ];

  test.each(cases)("%s, %s", (mode, source, label) => {
    expect(getWorkflowSourceLabel(source, mode).label).toBe(label);
  });
});
