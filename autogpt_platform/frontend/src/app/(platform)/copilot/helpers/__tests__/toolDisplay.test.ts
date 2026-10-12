import { readUIMessageStream, type UIMessage, type UIMessageChunk } from "ai";
import { describe, expect, it } from "vitest";
import {
  getChainHeading,
  toChainRow,
} from "../../components/ToolChain/helpers";
import { getSessionActivity } from "../../components/WorkspaceFileCards/helpers";
import { convertChatSessionMessagesToUiMessages } from "../convertChatSessionToUiMessages";
import { getAgentDisplayName, withToolDisplayNames } from "../toolDisplay";

type Part = UIMessage["parts"][number];

function runPart(toolCallId = "call-one"): Part {
  return {
    type: "tool-run_agent",
    toolCallId,
    state: "input-available",
    input: { library_agent_id: "b71fd24c-7623-4a73-a000-000000000000" },
  };
}

function displayPart(toolCallId: string, displayName: unknown): Part {
  return {
    type: "data-tool-display",
    id: toolCallId,
    data: { toolCallId, displayName },
  };
}

describe("agent display names", () => {
  it("updates concurrent tool rows and the live heading by actual call ID", () => {
    const first = runPart();
    const second = runPart("call-two");
    const parts = withToolDisplayNames([
      first,
      displayPart("call-two", "Second workflow"),
      second,
      displayPart("call-one", "Daily briefing"),
    ]);
    const rows = parts.flatMap((part, index) => toChainRow(part, index) ?? []);
    expect(rows.map((row) => row.text)).toEqual([
      'Running agent "Daily briefing"…',
      'Running agent "Second workflow"…',
    ]);
    expect(getChainHeading(rows, true)).toBe(
      'Running agent "Second workflow"…',
    );
    expect(parts[0]).toMatchObject({
      input: { library_agent_id: "b71fd24c-7623-4a73-a000-000000000000" },
    });
    expect(first).not.toHaveProperty("title");
    expect(
      getSessionActivity([{ id: "m", role: "assistant", parts }]).runs.map(
        (run) => run.name,
      ),
    ).toEqual(["Second workflow", "Daily briefing"]);
  });

  it("shows a generic label for unresolved IDs, slugs and streaming input", () => {
    for (const input of [
      { library_agent_id: "b71fd24c-7623-4a73-a000-000000000000" },
      { username_agent_slug: "creator/daily-briefing" },
      { agent_name: "Untrusted argument" },
    ]) {
      expect(toChainRow({ ...runPart(), input } as Part, 0)?.text).toBe(
        "Running agent…",
      );
      expect(
        toChainRow({ ...runPart(), input, state: "input-streaming" } as Part, 0)
          ?.text,
      ).toBe("Running agent…");
    }
  });

  it.each([
    [{ graph_name: "Graph name" }, "Graph name"],
    [
      JSON.stringify({ type: "agent_output", agent_name: "Waited run" }),
      "Waited run",
    ],
    [
      { type: "agent_details", agent: { name: "Agent details" } },
      "Agent details",
    ],
    [
      { type: "setup_requirements", setup_info: { agent_name: "Setup name" } },
      "Setup name",
    ],
    [{ name: "Unrelated name", outputs: { agent_name: "Output value" } }, null],
    [{ agent: { name: "Untyped nested name" } }, null],
    ["{truncated", null],
    [{ graph_name: "  ", agent_name: 42 }, null],
  ])(
    "resolves only supported historical result names from %j",
    (output, expected) => {
      expect(getAgentDisplayName(undefined, output)).toBe(expected);
    },
  );

  it.each([undefined, null, "", "  ", 42, {}, []])(
    "ignores malformed display names %j",
    (displayName) => {
      const parts = withToolDisplayNames([
        runPart(),
        displayPart("call-one", displayName),
      ]);
      expect(toChainRow(parts[0], 0)?.text).toBe("Running agent…");
      expect(
        getAgentDisplayName(displayName, { graph_name: "Result fallback" }),
      ).toBe("Result fallback");
    },
  );

  it("keeps the latest valid name on replay and uses it for done and error labels", () => {
    const parts = withToolDisplayNames([
      runPart(),
      displayPart("call-one", "First name"),
      displayPart("call-one", " Canonical name "),
      displayPart("call-one", "  "),
      { type: "data-tool-display", data: { displayName: "No call ID" } },
    ]);
    expect(
      toChainRow(
        {
          ...parts[0],
          state: "output-available",
          output: { graph_name: "Older name" },
        } as Part,
        0,
      )?.text,
    ).toBe('Ran agent "Canonical name"');
    expect(
      toChainRow(
        {
          ...parts[0],
          state: "output-error",
          errorText: "Execution failed",
        } as Part,
        0,
      )?.text,
    ).toBe('Failed while running agent "Canonical name"');
  });

  it.each([
    { graph_name: "Historical workflow" },
    JSON.stringify({ agent_name: "Historical workflow" }),
  ])(
    "recovers names from historical object or JSON outputs across pages",
    (output) => {
      const { messages } = convertChatSessionMessagesToUiMessages(
        "session",
        [
          {
            role: "assistant",
            sequence: 1,
            tool_calls: [
              {
                id: "call-one",
                type: "function",
                function: {
                  name: "run_agent",
                  arguments: { library_agent_id: "library-id" },
                },
              },
            ],
          },
        ],
        { extraToolOutputs: new Map([["call-one", output]]) },
      );
      expect(toChainRow(messages[0].parts[0], 0)?.text).toBe(
        'Ran agent "Historical workflow"',
      );
    },
  );

  it.each([false, true])(
    "hydrates persisted names for active/settled sessions and public shares (complete=%s)",
    (isComplete) => {
      const { messages } = convertChatSessionMessagesToUiMessages(
        "session",
        [
          {
            role: "assistant",
            sequence: 1,
            tool_calls: [
              {
                id: "call-one",
                type: "function",
                display_name: "Daily briefing",
                function: {
                  name: "run_agent",
                  arguments: '{"library_agent_id":"library-id"}',
                },
              },
            ],
          },
        ],
        {
          isComplete,
          fileUrlBuilder: (id) =>
            `/api/public/shared/chats/token/files/${id}/download`,
        },
      );
      expect(messages[0].parts[0]).toMatchObject({
        title: "Daily briefing",
        input: { library_agent_id: "library-id" },
      });
      expect(toChainRow(messages[0].parts[0], 0)?.text).toBe(
        isComplete
          ? 'Ran agent "Daily briefing"'
          : 'Running agent "Daily briefing"…',
      );
    },
  );

  it("consumes non-transient AI SDK display parts and upserts them during stream replay", async () => {
    const chunks: UIMessageChunk[] = [
      { type: "start", messageId: "message" },
      {
        type: "tool-input-available",
        toolCallId: "call-one",
        toolName: "run_agent",
        input: { library_agent_id: "library-id" },
      },
      {
        type: "data-tool-display",
        id: "call-one",
        data: { toolCallId: "call-one", displayName: "Daily briefing" },
      },
      {
        type: "data-tool-display",
        id: "call-one",
        data: { toolCallId: "call-one", displayName: "Daily briefing updated" },
      },
      {
        type: "tool-output-available",
        toolCallId: "call-one",
        output: { graph_name: "Old result name" },
      },
      { type: "finish" },
    ];
    const stream = new ReadableStream<UIMessageChunk>({
      start(controller) {
        chunks.forEach((chunk) => controller.enqueue(chunk));
        controller.close();
      },
    });
    let latest: UIMessage | undefined;
    for await (const message of readUIMessageStream({ stream }))
      latest = message;
    expect(
      latest?.parts.filter((part) => part.type === "data-tool-display"),
    ).toHaveLength(1);
    const parts = withToolDisplayNames(latest?.parts ?? []);
    expect(toChainRow(parts[0], 0)?.text).toBe(
      'Ran agent "Daily briefing updated"',
    );
  });
});
