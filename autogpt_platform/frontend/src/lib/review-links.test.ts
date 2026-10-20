import { describe, expect, it } from "vitest";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { getReviewLink } from "./review-links";

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
    instructions: "Approve outreach email",
    editable: true,
    status: "WAITING",
    created_at: new Date(),
    ...overrides,
  };
}

describe("getReviewLink", () => {
  it("opens the CoPilot thread for a session-scoped review", () => {
    expect(getReviewLink(makeReview({ session_id: "sess-1" }))).toBe(
      "/copilot?sessionId=sess-1",
    );
  });

  it("selects the run on the library agent page", () => {
    // activeTab/activeItem are the params NewAgentLibraryView parses;
    // anything else lands on the agent page without opening the run.
    expect(
      getReviewLink(
        makeReview({ library_agent_id: "lib-1", graph_exec_id: "run-1" }),
      ),
    ).toBe("/library/agents/lib-1?activeTab=runs&activeItem=run-1");
  });

  it("prefers the session link when both ids are present", () => {
    expect(
      getReviewLink(
        makeReview({ session_id: "sess-1", library_agent_id: "lib-1" }),
      ),
    ).toBe("/copilot?sessionId=sess-1");
  });

  it("falls back to the library when neither id resolved", () => {
    expect(getReviewLink(makeReview())).toBe("/library");
  });
});
