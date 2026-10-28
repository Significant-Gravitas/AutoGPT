import { describe, expect, it } from "vitest";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { getReviewTitle } from "./helpers";

function makeReview(
  overrides: Partial<PendingHumanReviewModel> = {},
): PendingHumanReviewModel {
  return {
    node_exec_id: "ne-1",
    node_id: "n-1",
    user_id: "u-1",
    graph_exec_id: "run-1",
    graph_id: "g-1",
    graph_version: 1,
    payload: {},
    editable: true,
    status: "WAITING",
    created_at: new Date(),
    ...overrides,
  };
}

describe("getReviewTitle", () => {
  it("prefers the review instructions", () => {
    expect(
      getReviewTitle(
        makeReview({
          instructions: "Approve email",
          agent_name: "Lead Finder",
        }),
      ),
    ).toBe("Approve email");
  });

  it("falls back to the agent name without instructions", () => {
    expect(getReviewTitle(makeReview({ agent_name: "Lead Finder" }))).toBe(
      "Lead Finder",
    );
  });

  it("falls back to a generic label when neither resolved", () => {
    expect(getReviewTitle(makeReview())).toBe("Review needed");
  });
});
