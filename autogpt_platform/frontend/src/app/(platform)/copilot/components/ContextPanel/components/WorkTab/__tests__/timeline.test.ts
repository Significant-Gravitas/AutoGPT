import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { describe, expect, it } from "vitest";
import type { ChatDelegation } from "../../../../../delegations";
import { buildTimeline } from "../timeline";

const DELEGATION = {
  prompt: "Draft the PRD for the onboarding revamp",
  approved: true,
  startedAt: "2026-09-28T10:42:00Z",
  finishedAt: null,
  error: null,
} as ChatDelegation;

function session(messages: Record<string, unknown>[]) {
  return { messages } as unknown as SessionDetailResponse;
}

const RUN = [
  { role: "user", content: "An older brief" },
  { role: "assistant", content: "Old answer" },
  {
    role: "user",
    content: "Draft the PRD for the onboarding revamp: goals, scope",
    created_at: "2026-09-28T10:42:00Z",
  },
  {
    role: "assistant",
    content: "Reading the research first.",
    created_at: "2026-09-28T10:43:00Z",
    tool_calls: [
      {
        id: "t1",
        function: {
          name: "ask_question",
          arguments: JSON.stringify({ questions: [{ question: "Q4?" }] }),
        },
      },
    ],
  },
  { role: "user", content: "Q4", created_at: "2026-09-28T10:46:00Z" },
  {
    role: "assistant",
    content: "PRD drafted.",
    created_at: "2026-09-28T10:48:00Z",
  },
];

describe("buildTimeline", () => {
  it("tells the run from the brief on, with kinds", () => {
    const entries = buildTimeline(session(RUN), DELEGATION, "completed");
    expect(entries.map((e) => [e.kind, e.text])).toEqual([
      ["Approved", "You approved the hand-off"],
      ["Thought", "Reading the research first."],
      ["Question", expect.any(String)],
      ["You answered", "Q4"],
      ["Response", "PRD drafted."],
    ]);
    expect(entries[1].at).toBe(Date.parse("2026-09-28T10:43:00Z"));
  });

  it("ends a failed run on its error", () => {
    const entries = buildTimeline(
      session([]),
      { ...DELEGATION, approved: false, error: "Weekly budget cap reached" },
      "failed",
    );
    expect(entries).toMatchObject([
      { kind: "Error", text: "Weekly budget cap reached" },
    ]);
  });
});
