import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";
import {
  countLiveDelegations,
  delegationActivityText,
  delegationTone,
  foldDelegations,
  formatElapsed,
} from "./helpers";

function toolPart(
  toolName: string,
  toolCallId: string,
  input: unknown,
  output?: unknown,
  state: "input-available" | "output-available" | "output-error" = output ===
  undefined
    ? "input-available"
    : "output-available",
  errorText?: string,
) {
  return {
    type: `tool-${toolName}`,
    toolCallId,
    state,
    input,
    output,
    errorText,
  } as UIMessage["parts"][number];
}

function assistant(id: string, parts: UIMessage["parts"]): UIMessage {
  return { id, role: "assistant", parts };
}

const EXPERT = {
  id: "exp-1",
  name: "Mira",
  role: "Researcher",
  avatar_url: "https://cdn/mira.png",
};

describe("foldDelegations", () => {
  it("ignores user messages and non-delegation tools", () => {
    const messages: UIMessage[] = [
      { id: "u", role: "user", parts: [{ type: "text", text: "hi" }] },
      assistant("a", [toolPart("web_search", "c1", {}, { results: [] })]),
    ];
    expect(foldDelegations(messages)).toEqual([]);
  });

  it("builds a pending row from the input while the delegate call is still running", () => {
    const rows = foldDelegations([
      assistant("a", [
        toolPart("delegate_to_expert", "c1", {
          expert_id: "exp-1",
          prompt: "Find three vendors",
        }),
      ]),
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      key: "c1",
      kind: "delegate",
      expertId: "exp-1",
      brief: "Find three vendors",
      status: "running",
      toolState: "running",
      subSessionId: null,
      link: null,
      runs: 1,
    });
  });

  it("keeps the row key stable once the output names the sub-session", () => {
    const pending = foldDelegations([
      assistant("a", [
        toolPart("delegate_to_expert", "c1", { expert_id: "exp-1" }),
      ]),
    ]);
    const settled = foldDelegations([
      assistant("a", [
        toolPart(
          "delegate_to_expert",
          "c1",
          { expert_id: "exp-1" },
          {
            status: "completed",
            sub_session_id: "sub-1",
            response: "Done: vendor list attached",
            elapsed_seconds: 42,
            expert: EXPERT,
            sub_autopilot_session_link: "/copilot?sessionId=sub-1&x=1",
          },
        ),
      ]),
    ]);
    expect(settled[0].key).toBe(pending[0].key);
    expect(settled[0]).toMatchObject({
      subSessionId: "sub-1",
      status: "completed",
      toolState: "done",
      response: "Done: vendor list attached",
      elapsedSeconds: 42,
      expertName: "Mira",
      expertRole: "Researcher",
      expertAvatarUrl: "https://cdn/mira.png",
      link: "/copilot?sessionId=sub-1&x=1",
    });
  });

  it("folds delegate → poll → poll into one row that updates in place", () => {
    const rows = foldDelegations([
      assistant("a", [
        toolPart(
          "delegate_to_expert",
          "c1",
          { expert_id: "exp-1", prompt: "Draft the brief" },
          { status: "running", sub_session_id: "sub-1", expert: EXPERT },
        ),
        toolPart(
          "get_sub_session_result",
          "c2",
          { sub_session_id: "sub-1" },
          {
            status: "running",
            sub_session_id: "sub-1",
            progress: {
              message_count: 4,
              last_messages: [
                { role: "user", content: "Draft the brief" },
                { role: "assistant", content: "Reading the sources now" },
              ],
            },
          },
        ),
        toolPart(
          "get_sub_session_result",
          "c3",
          { sub_session_id: "sub-1" },
          {
            status: "completed",
            sub_session_id: "sub-1",
            response: "Brief drafted",
            elapsed_seconds: 90,
          },
        ),
      ]),
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      key: "c1",
      status: "completed",
      response: "Brief drafted",
      lastMessage: "Reading the sources now",
      brief: "Draft the brief",
      elapsedSeconds: 90,
      runs: 1,
    });
  });

  it("counts a re-delegation of the same sub-session as another run and clears the old answer", () => {
    const rows = foldDelegations([
      assistant("a", [
        toolPart(
          "delegate_to_expert",
          "c1",
          { expert_id: "exp-1", prompt: "First task" },
          {
            status: "completed",
            sub_session_id: "sub-1",
            response: "First answer",
          },
        ),
      ]),
      assistant("b", [
        toolPart(
          "delegate_to_expert",
          "c2",
          { expert_id: "exp-1", prompt: "Second task" },
          { status: "running", sub_session_id: "sub-1" },
        ),
      ]),
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      key: "c1",
      runs: 2,
      brief: "Second task",
      status: "running",
      response: null,
    });
  });

  it("keeps rows in first-seen order across several experts", () => {
    const rows = foldDelegations([
      assistant("a", [
        toolPart(
          "handoff_to_expert",
          "c1",
          { expert_id: "exp-2", prompt: "Own this" },
          { status: "transferred", sub_session_id: "sub-2" },
        ),
        toolPart(
          "run_sub_session",
          "c2",
          { prompt: "Crunch numbers" },
          { status: "completed", sub_session_id: "sub-3" },
        ),
      ]),
    ]);
    expect(rows.map((row) => [row.key, row.kind, row.status])).toEqual([
      ["c1", "handoff", "transferred"],
      ["c2", "sub_session", "completed"],
    ]);
    expect(rows[0].link).toBe("/copilot?sessionId=sub-2");
  });

  it("creates a row from a poll whose start call is outside the loaded history", () => {
    const rows = foldDelegations([
      assistant("a", [
        toolPart(
          "get_sub_session_result",
          "c9",
          { sub_session_id: "sub-old" },
          {
            status: "running",
            sub_session_id: "sub-old",
            expert: EXPERT,
          },
        ),
      ]),
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      key: "c9",
      kind: "sub_session",
      subSessionId: "sub-old",
      expertName: "Mira",
      runs: 0,
    });
  });

  it("ignores a poll with no sub-session id and no known row", () => {
    const rows = foldDelegations([
      assistant("a", [
        toolPart("get_sub_session_result", "c1", {}, { status: "error" }),
      ]),
    ]);
    expect(rows).toEqual([]);
  });

  it("marks a failed tool call as an error row with the error text", () => {
    const rows = foldDelegations([
      assistant("a", [
        toolPart(
          "delegate_to_expert",
          "c1",
          { expert_id: "exp-1" },
          undefined,
          "output-error",
          "Expert not found",
        ),
      ]),
    ]);
    expect(rows[0]).toMatchObject({
      status: "error",
      toolState: "error",
      errorText: "Expert not found",
    });
    expect(countLiveDelegations(rows)).toBe(0);
  });
});

describe("countLiveDelegations", () => {
  it("counts running tool calls and frozen live statuses, not settled ones", () => {
    const rows = foldDelegations([
      assistant("a", [
        toolPart("delegate_to_expert", "c1", { expert_id: "exp-1" }),
        toolPart(
          "delegate_to_expert",
          "c2",
          { expert_id: "exp-2" },
          { status: "queued", sub_session_id: "sub-2" },
        ),
        toolPart(
          "delegate_to_expert",
          "c3",
          { expert_id: "exp-3" },
          { status: "completed", sub_session_id: "sub-3" },
        ),
      ]),
    ]);
    expect(countLiveDelegations(rows)).toBe(2);
  });
});

describe("delegationActivityText", () => {
  const base = foldDelegations([
    assistant("a", [
      toolPart(
        "delegate_to_expert",
        "c1",
        { expert_id: "exp-1", prompt: "The brief" },
        {
          status: "completed",
          sub_session_id: "sub-1",
          response: "The answer",
          progress: { last_messages: [{ role: "assistant", content: "Step" }] },
        },
      ),
    ]),
  ])[0];

  it("leads with the latest step while live and the answer once settled", () => {
    expect(delegationActivityText(base, "running")).toBe("Step");
    expect(delegationActivityText(base, "completed")).toBe("The answer");
  });

  it("shows the brief for a hand-off since the caller never gets an answer", () => {
    expect(
      delegationActivityText({ ...base, kind: "handoff" }, "transferred"),
    ).toBe("The brief");
  });

  it("puts the error first on failed rows", () => {
    expect(
      delegationActivityText({ ...base, errorText: "Boom" }, "error"),
    ).toBe("Boom");
  });
});

describe("delegationTone / formatElapsed", () => {
  it("maps statuses to tones", () => {
    expect(delegationTone("running")).toBe("working");
    expect(delegationTone("QUEUED")).toBe("working");
    expect(delegationTone("completed")).toBe("done");
    expect(delegationTone("transferred")).toBe("done");
    expect(delegationTone("error")).toBe("failed");
    expect(delegationTone("cancelled")).toBe("stopped");
    expect(delegationTone("unknown")).toBe("stopped");
  });

  it("formats seconds, minutes and hours", () => {
    expect(formatElapsed(42)).toBe("42s");
    expect(formatElapsed(125)).toBe("2m 05s");
    expect(formatElapsed(3720)).toBe("1h 02m");
  });
});
