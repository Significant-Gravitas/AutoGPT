import type { DelegationSummary } from "@/app/api/__generated__/models/delegationSummary";
import { describe, expect, it } from "vitest";
import {
  filterDelegations,
  formatDelegationDay,
  formatDelegationTime,
  getExpertSummaryLine,
  getHandoffMeta,
  getOttoSummaryLine,
} from "../helpers";

const NOW = new Date(2026, 8, 28, 12, 0);

function at(daysBack: number, hour = 10, minute = 41) {
  return new Date(2026, 8, 28 - daysBack, hour, minute);
}

function delegation(over: Partial<DelegationSummary>): DelegationSummary {
  return {
    sub_session_id: "sub",
    parent_session_id: "parent",
    expert: { id: "exp-alex", name: "Alex", role: "PM" },
    title: "PRD",
    brief: "",
    status: "completed",
    created_at: at(0),
    finished_at: null,
    elapsed_seconds: 400,
    cost_usd: 0.31,
    files_count: 1,
    question: null,
    question_options: [],
    ...over,
  };
}

describe("delegation list helpers", () => {
  it("labels today by clock, then Yesterday, then the weekday", () => {
    expect(formatDelegationTime(at(0), NOW)).toBe("10:41");
    expect(formatDelegationTime(at(1), NOW)).toBe("Yesterday");
    expect(formatDelegationTime(at(3), NOW)).toBe(
      at(3).toLocaleDateString([], { weekday: "short" }),
    );
    expect(formatDelegationDay(at(0), NOW)).toBe("today 10:41");
    expect(formatDelegationDay(at(1), NOW)).toBe("yesterday");
  });

  it("groups proposed and needs_input as needing you, cancelled as failed", () => {
    const rows = (
      ["proposed", "needs_input", "queued", "running", "cancelled"] as const
    ).map((status) => delegation({ status, sub_session_id: status }));
    expect(filterDelegations(rows, "needs_you").map((d) => d.status)).toEqual([
      "proposed",
      "needs_input",
    ]);
    expect(filterDelegations(rows, "working").map((d) => d.status)).toEqual([
      "queued",
      "running",
    ]);
    expect(filterDelegations(rows, "failed").map((d) => d.status)).toEqual([
      "cancelled",
    ]);
  });

  it("writes the expert's meta line from Otto", () => {
    expect(getHandoffMeta(delegation({}), NOW)).toBe(
      "From Otto · today 10:41 · 6m 40s · $0.31 · 1 file",
    );
  });

  it("sums up Otto's day and an expert's week", () => {
    expect(
      getOttoSummaryLine(3, {
        working: 2,
        needs_you: 1,
        completed: 0,
        failed: 0,
        spent_today_usd: 0.4,
      }),
    ).toBe("3 delegations today · 2 working · 1 needs you · $0.40 spent");
    expect(
      getExpertSummaryLine(
        [
          delegation({ cost_usd: 1 }),
          delegation({ status: "failed", cost_usd: 0.67 }),
          delegation({ created_at: at(9) }),
        ],
        NOW,
      ),
    ).toBe("2 delegations this week · 1 done · 1 failed · $1.67 spent");
  });
});
