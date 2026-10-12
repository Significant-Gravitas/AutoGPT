import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";
import {
  getChainHeading,
  toChainRow,
} from "../../components/ToolChain/helpers";
import { convertChatSessionMessagesToUiMessages } from "../convertChatSessionToUiMessages";
import {
  getAgentDisplayName,
  getBlockDisplayName,
  withToolDisplayNames,
} from "../toolDisplay";

type Part = UIMessage["parts"][number];
const blockID = "db7d8f02-2f44-4c55-ab7a-eae0941f0c30";

function blockPart(
  tool = "run_block",
  extras: Record<string, unknown> = {},
): Part {
  return {
    type: `tool-${tool}`,
    toolCallId: "block-call",
    state: "input-available",
    input: { block_id: blockID },
    ...extras,
  } as Part;
}

describe("block display names", () => {
  it.each(["run_block", "continue_run_block"])(
    "renders canonical %s names in live rows and the heading",
    (tool) => {
      const input = { block_id: blockID };
      const parts = withToolDisplayNames([
        blockPart(tool, { input }),
        {
          type: "data-tool-display",
          id: "block-call",
          data: {
            toolCallId: "block-call",
            displayName: "FillTextTemplateBlock",
          },
        },
      ]);
      const row = toChainRow(parts[0], 0)!;
      expect(row.text).toBe(
        tool === "run_block"
          ? 'Running block "Fill Text Template"…'
          : 'Continuing block run "Fill Text Template"…',
      );
      expect(getChainHeading([row], true)).toBe(row.text);
      expect(row.input).toEqual(input);
      expect(row.text).not.toContain(blockID);
    },
  );

  it.each([undefined, "", "  ", 12, {}, []])(
    "ignores malformed metadata %j and recovers historical names",
    (name) => {
      expect(
        getBlockDisplayName(name, { block_name: "FillTextTemplateBlock" }),
      ).toBe("Fill Text Template");
    },
  );

  it.each([
    [{ block_name: "FillTextTemplateBlock" }, "Fill Text Template"],
    ['{"block":{"name":"HTTPAPIRequestBlock"}}', "HTTPAPI Request"],
    [
      {
        type: "setup_requirements",
        setup_info: { agent_name: "FillTextTemplateBlock" },
      },
      "Fill Text Template",
    ],
    [{ block_name: "  ", block: { name: "TextBlock" } }, "Text"],
    [{ block_id: blockID, outputs: { name: "unrelated output" } }, null],
    ["{truncated", null],
  ])("reads supported legacy result names from %j", (output, expected) => {
    expect(getBlockDisplayName(undefined, output)).toBe(expected);
  });

  it("gives metadata priority and preserves human agent names unchanged", () => {
    expect(
      getBlockDisplayName("FillTextTemplateBlock", { block_name: "OldBlock" }),
    ).toBe("Fill Text Template");
    expect(getAgentDisplayName("Daily briefing Block")).toBe(
      "Daily briefing Block",
    );
  });

  it.each(["run_block", "continue_run_block"])(
    "keeps unresolved %s and streaming input generic",
    (tool) => {
      for (const state of ["input-streaming", "input-available"]) {
        const row = toChainRow(blockPart(tool, { state }), 0)!;
        expect(row.text).toBe(
          tool === "run_block" ? "Running block…" : "Continuing block run…",
        );
        expect(row.text).not.toContain(blockID);
      }
    },
  );

  it.each([
    [
      {
        state: "output-available",
        output: { block_name: "FillTextTemplateBlock" },
      },
      'Ran block "Fill Text Template"',
    ],
    [
      {
        state: "output-error",
        title: "FillTextTemplateBlock",
        errorText: "Failed",
      },
      'Failed while running block "Fill Text Template"',
    ],
    [
      {
        state: "output-available",
        output: {
          type: "review_required",
          block_name: "FillTextTemplateBlock",
        },
      },
      "Review Fill Text Template",
    ],
    [
      {
        state: "output-available",
        output: {
          type: "setup_requirements",
          setup_info: { agent_name: "FillTextTemplateBlock" },
        },
      },
      "Connect Fill Text Template to continue",
    ],
  ])(
    "normalizes completed, failed and action-required rows %j",
    (extras, expected) => {
      expect(toChainRow(blockPart("run_block", extras), 0)?.text).toBe(
        expected,
      );
    },
  );

  it.each(["run_block", "continue_run_block"])(
    "retains %s names on reloaded/shared history",
    (tool) => {
      const { messages } = convertChatSessionMessagesToUiMessages(
        "session",
        [
          {
            role: "assistant",
            sequence: 1,
            tool_calls: [
              {
                id: "block-call",
                type: "function",
                display_name: "FillTextTemplateBlock",
                function: {
                  name: tool,
                  arguments: JSON.stringify({ block_id: blockID }),
                },
              },
            ],
          },
        ],
        { isComplete: true },
      );
      expect(toChainRow(messages[0].parts[0], 0)?.text).toBe(
        tool === "run_block"
          ? 'Ran block "Fill Text Template"'
          : 'Continued block run "Fill Text Template"',
      );
    },
  );
});
