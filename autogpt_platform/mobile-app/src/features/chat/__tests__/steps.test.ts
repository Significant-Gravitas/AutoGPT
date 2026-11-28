import type { UIMessage } from "ai";

import { groupParts, stepKind, stepLabel, type StepPart } from "../steps";

type Parts = UIMessage["parts"];

function tool(
  id: string,
  name = "web_search",
  extra: Partial<StepPart> = {},
): StepPart {
  return {
    type: `tool-${name}`,
    toolCallId: id,
    state: "output-available",
    input: {},
    output: {},
    ...extra,
  } as StepPart;
}

describe("stepKind", () => {
  it.each([
    ["TodoWrite", "hidden"],
    ["read_workspace_file", "file"],
    ["web_search", "web"],
    ["schedule_routine", "routine"],
    ["memory_search", "memory"],
    ["bash_exec", "computer"],
    ["connect_integration", "connect"],
    ["list_team", "team"],
    ["ask_question", "message"],
    ["find_library_agent", "workflow"],
    ["something_else", "other"],
  ])("maps %s to %s", (name, kind) => {
    expect(stepKind(name)).toBe(kind);
  });
});

describe("stepLabel", () => {
  it("falls back when there is no title", () => {
    expect(stepLabel(tool("a"))).toBe("Working on it");
  });

  it("adds a suffix to a failed step", () => {
    const part = tool("a", "web_search", {
      title: "Searched the web",
      state: "output-error",
      errorText: "boom",
    } as Partial<StepPart>);
    expect(stepLabel(part)).toBe("Searched the web · didn't finish");
  });
});

describe("groupParts", () => {
  it("joins tools across step-start parts into one block", () => {
    const parts = [
      tool("a"),
      { type: "step-start" },
      tool("b"),
      { type: "text", text: "Done", state: "done" },
    ] as Parts;

    const blocks = groupParts(parts);

    expect(blocks.map((block) => block.kind)).toEqual(["steps", "text"]);
    expect(blocks[0].kind === "steps" && blocks[0].steps).toHaveLength(2);
    expect(blocks[0].key).toBe("a");
  });

  it("drops a run made only of hidden tools", () => {
    expect(groupParts([tool("a", "TodoWrite")] as Parts)).toEqual([]);
  });

  it("splits tools on either side of text into two blocks", () => {
    const parts = [
      tool("a"),
      { type: "text", text: "Halfway", state: "done" },
      tool("b"),
    ] as Parts;

    expect(groupParts(parts).map((block) => block.kind)).toEqual([
      "steps",
      "text",
      "steps",
    ]);
  });

  it("gives the onboarding card a block of its own", () => {
    const parts = [tool("a"), tool("b", "expert_onboarding"), tool("c")] as Parts;

    const blocks = groupParts(parts);

    expect(blocks.map((block) => block.kind)).toEqual([
      "steps",
      "onboarding",
      "steps",
    ]);
    expect(blocks[1].key).toBe("b");
  });

  it("turns a result asking for a service into a setup card", () => {
    const setupOutput = {
      type: "setup_requirements",
      message: "To continue, please connect your GitHub account.",
      setup_info: {
        agent_id: "connect_github",
        agent_name: "GitHub",
        user_readiness: {
          missing_credentials: {
            github_credentials: { provider: "github", types: ["oauth2"] },
          },
        },
      },
    };
    const parts = [
      tool("a"),
      tool("b", "connect_integration", { output: setupOutput } as Partial<StepPart>),
      tool("c"),
    ] as Parts;

    const blocks = groupParts(parts);

    expect(blocks.map((block) => block.kind)).toEqual(["steps", "setup", "steps"]);
    expect(blocks[1].key).toBe("b");
  });

  it("keeps a setup result with nothing missing, or still running, as a step", () => {
    const connected = {
      type: "setup_requirements",
      setup_info: { user_readiness: { missing_credentials: {} } },
    };
    const parts = [
      tool("a", "run_mcp_tool", { output: connected } as Partial<StepPart>),
      tool("b", "connect_integration", { state: "input-available" } as Partial<StepPart>),
    ] as Parts;

    const blocks = groupParts(parts);

    expect(blocks.map((block) => block.kind)).toEqual(["steps"]);
    expect(blocks[0].kind === "steps" && blocks[0].steps).toHaveLength(2);
  });
});
