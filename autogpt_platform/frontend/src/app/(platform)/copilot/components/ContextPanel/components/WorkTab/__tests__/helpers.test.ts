import { describe, expect, it } from "vitest";
import type { ChatDelegation } from "../../../../../delegations";
import { isBudgetError, workTotals } from "../helpers";

describe("isBudgetError", () => {
  it.each([
    "Weekly budget cap reached ($5.00)",
    "Delegation cap reached",
    "Spend limit hit for this chat",
    "The cost cap was exceeded",
    "Cap reached before the draft finished",
  ])("offers raising the budget for %j", (error) => {
    expect(isBudgetError(error)).toBe(true);
  });

  it.each([
    "Rate limit exceeded, try again later",
    "Context length limit reached",
    "Wrote a recap of the meeting",
    "Could not spend time on it",
    "",
  ])("does not for %j", (error) => {
    expect(isBudgetError(error)).toBe(false);
  });

  it("does not without an error", () => {
    expect(isBudgetError(null)).toBe(false);
  });
});

describe("workTotals", () => {
  const base: ChatDelegation = {
    toolCallId: "c1",
    tool: "delegate_to_expert",
    expertId: "exp-alex",
    expert: null,
    prompt: null,
    subSessionId: "sub-1",
    link: null,
    elapsedSeconds: null,
    costUsd: null,
    startedAt: null,
    finishedAt: null,
    response: null,
    question: null,
    questionOptions: [],
    error: null,
    files: [],
    reviewId: null,
    approved: false,
    superseded: false,
    status: "running",
  };

  it("counts a re-delegated thread and its spend once", () => {
    const totals = workTotals(
      [
        { ...base, toolCallId: "c1", costUsd: 0.2, superseded: true },
        { ...base, toolCallId: "c2", costUsd: 0.5 },
        { ...base, toolCallId: "c3", subSessionId: "sub-2", costUsd: 0.25 },
      ],
      (d) => d.status,
    );
    expect(totals).toEqual([
      { label: "Working", value: "2" },
      { label: "Waiting", value: "0" },
      { label: "Spent", value: "$0.75" },
    ]);
  });
});
