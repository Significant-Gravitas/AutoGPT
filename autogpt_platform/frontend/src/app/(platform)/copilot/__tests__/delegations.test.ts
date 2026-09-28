import { describe, expect, it } from "vitest";
import type { UIMessage } from "ai";
import {
  countDelegations,
  formatElapsed,
  getChatDelegations,
  getDelegationSummary,
} from "../delegations";

function toolPart(
  tool: string,
  toolCallId: string,
  input: unknown,
  output?: unknown,
  state: string = output === undefined ? "input-available" : "output-available",
) {
  return { type: `tool-${tool}`, state, toolCallId, input, output };
}

function assistant(id: string, parts: unknown[]): UIMessage {
  return { id, role: "assistant", parts } as unknown as UIMessage;
}

const ALEX = {
  id: "exp-alex",
  name: "Alex",
  role: "Product Manager",
  avatar_url: "/alex.png",
  color: "#abc",
};

describe("getChatDelegations", () => {
  it("opens a running delegation from a blocking call with no output yet", () => {
    const messages = [
      assistant("m1", [
        toolPart("delegate_to_expert", "call-1", {
          expert_id: "exp-alex",
          prompt: "Draft the PRD",
        }),
      ]),
    ];
    const [delegation] = getChatDelegations(messages);
    expect(delegation).toMatchObject({
      toolCallId: "call-1",
      tool: "delegate_to_expert",
      expertId: "exp-alex",
      prompt: "Draft the PRD",
      status: "running",
      subSessionId: null,
    });
  });

  it("reads the expert, run id and result off the tool output", () => {
    const messages = [
      assistant("m1", [
        toolPart(
          "delegate_to_expert",
          "call-1",
          { expert_id: "exp-alex", prompt: "Draft the PRD" },
          {
            type: "mcp_tool_output",
            status: "completed",
            sub_session_id: "sub-1",
            sub_autopilot_session_link: "/copilot?sessionId=sub-1",
            elapsed_seconds: 400,
            response: "Draft attached.",
            expert: ALEX,
            sub_workspace_files: [
              { name: "prd.md", path: "/sessions/sub-1/prd.md" },
            ],
          },
        ),
      ]),
    ];
    const [delegation] = getChatDelegations(messages);
    expect(delegation).toMatchObject({
      status: "completed",
      subSessionId: "sub-1",
      link: "/copilot?sessionId=sub-1",
      elapsedSeconds: 400,
      response: "Draft attached.",
      expert: {
        id: "exp-alex",
        name: "Alex",
        role: "Product Manager",
        avatarUrl: "/alex.png",
      },
      files: [{ name: "prd.md", path: "/sessions/sub-1/prd.md" }],
    });
  });

  it("folds later polls of the same run into the delegation that opened it", () => {
    const messages = [
      assistant("m1", [
        toolPart(
          "delegate_to_expert",
          "call-1",
          { expert_id: "exp-alex", prompt: "Draft the PRD" },
          { status: "running", sub_session_id: "sub-1", expert: ALEX },
        ),
      ]),
      assistant("m2", [
        toolPart(
          "get_sub_session_result",
          "call-2",
          { sub_session_id: "sub-1" },
          {
            status: "completed",
            sub_session_id: "sub-1",
            elapsed_seconds: 90,
            response: "Done.",
          },
        ),
      ]),
    ];
    const delegations = getChatDelegations(messages);
    expect(delegations).toHaveLength(1);
    expect(delegations[0]).toMatchObject({
      status: "completed",
      elapsedSeconds: 90,
      response: "Done.",
      expert: { name: "Alex" },
    });
  });

  it("keeps a re-delegation to the same run as its own entry", () => {
    const first = {
      status: "completed",
      sub_session_id: "sub-1",
      response: "First answer",
      expert: ALEX,
    };
    const messages = [
      assistant("m1", [
        toolPart(
          "delegate_to_expert",
          "call-1",
          { expert_id: "exp-alex" },
          first,
        ),
        toolPart(
          "delegate_to_expert",
          "call-2",
          { expert_id: "exp-alex", prompt: "Again" },
          { status: "running", sub_session_id: "sub-1", expert: ALEX },
        ),
        toolPart(
          "get_sub_session_result",
          "call-3",
          { sub_session_id: "sub-1" },
          { status: "completed", sub_session_id: "sub-1", response: "Second" },
        ),
      ]),
    ];
    const delegations = getChatDelegations(messages);
    expect(delegations.map((d) => d.response)).toEqual([
      "First answer",
      "Second",
    ]);
  });

  it("marks a held hand-off as proposed with its review id", () => {
    const messages = [
      assistant("m1", [
        toolPart(
          "delegate_to_expert",
          "call-1",
          { expert_id: "exp-alex", prompt: "Draft the PRD" },
          { type: "approval_required", review_id: "rev-1", ask: "Hand off" },
        ),
      ]),
    ];
    const [delegation] = getChatDelegations(messages);
    expect(delegation.status).toBe("proposed");
    expect(delegation.reviewId).toBe("rev-1");
  });

  it("maps errors, cancellations and transfers", () => {
    const messages = [
      assistant("m1", [
        toolPart(
          "delegate_to_expert",
          "call-1",
          {},
          { type: "error", error: "Budget cap reached" },
        ),
        toolPart(
          "delegate_to_expert",
          "call-2",
          {},
          { status: "cancelled", sub_session_id: "sub-2" },
        ),
        toolPart(
          "handoff_to_expert",
          "call-3",
          {},
          { status: "transferred", sub_session_id: "sub-3" },
        ),
        toolPart("delegate_to_expert", "call-4", {}, undefined, "output-error"),
      ]),
    ];
    const statuses = getChatDelegations(messages).map((d) => d.status);
    expect(statuses).toEqual(["failed", "cancelled", "transferred", "failed"]);
    expect(getChatDelegations(messages)[0].error).toBe("Budget cap reached");
  });

  it("ignores user messages and unrelated tools", () => {
    const messages = [
      { id: "u", role: "user", parts: [{ type: "text", text: "hi" }] },
      assistant("m1", [
        toolPart("web_search", "call-1", { query: "x" }, { results: [] }),
        toolPart(
          "get_sub_session_result",
          "call-2",
          { sub_session_id: "orphan" },
          { status: "completed", sub_session_id: "orphan" },
        ),
      ]),
    ] as unknown as UIMessage[];
    expect(getChatDelegations(messages)).toEqual([]);
  });
});

describe("countDelegations / getDelegationSummary", () => {
  it("counts by status and words the summary most urgent first", () => {
    const messages = [
      assistant("m1", [
        toolPart("delegate_to_expert", "c1", {}, { type: "approval_required" }),
        toolPart("delegate_to_expert", "c2", {}, { status: "running" }),
        toolPart("delegate_to_expert", "c3", {}, { status: "queued" }),
        toolPart("delegate_to_expert", "c4", {}, { status: "completed" }),
      ]),
    ];
    const counts = countDelegations(getChatDelegations(messages));
    expect(counts).toMatchObject({
      proposed: 1,
      working: 1,
      queued: 1,
      done: 1,
      failed: 0,
      total: 4,
    });
    expect(getDelegationSummary(counts)).toBe(
      "1 hand-off waiting for your approval · 1 expert working · 1 queued",
    );
  });

  it("says stopped only when nothing is in flight", () => {
    const failedOnly = countDelegations(
      getChatDelegations([
        assistant("m1", [
          toolPart("delegate_to_expert", "c1", {}, { status: "error" }),
        ]),
      ]),
    );
    expect(getDelegationSummary(failedOnly)).toBe("1 expert stopped");
    const doneOnly = countDelegations(
      getChatDelegations([
        assistant("m1", [
          toolPart("delegate_to_expert", "c1", {}, { status: "completed" }),
        ]),
      ]),
    );
    expect(getDelegationSummary(doneOnly)).toBeNull();
  });
});

describe("formatElapsed", () => {
  it("formats seconds, minutes and hours", () => {
    expect(formatElapsed(null)).toBeNull();
    expect(formatElapsed(42)).toBe("42s");
    expect(formatElapsed(134)).toBe("2m 14s");
    expect(formatElapsed(3700)).toBe("1h 1m");
  });
});
