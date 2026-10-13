import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import { ChainRowView } from "../ChainRowView";
import type { ChainRow } from "../helpers";

const EMAIL = "Quarterly numbers: revenue is down 12%";

function row(overrides: Partial<ChainRow>): ChainRow {
  return {
    key: "row-1",
    category: "other",
    text: "Ran Gmail Read",
    state: "done",
    ...overrides,
  };
}

function gmailOutput() {
  return {
    type: "block_output",
    message: "Block executed",
    block_id: "25310c70-b89b-43ba-b25c-4dfa7e2a481c",
    block_name: "Gmail Read",
    provider: "google",
    success: true,
    outputs: { email: [EMAIL] },
  };
}

function expectHiddenFromReplay(text: string) {
  const element = screen.getByText(text, { exact: false });
  expect(element.closest(".ph-no-capture")).not.toBeNull();
  expect(element.closest(".sentry-block")).not.toBeNull();
}

describe("ChainRowView - session replay privacy", () => {
  it("keeps a Google block's output out of session replays", () => {
    render(
      <ChainRowView
        row={row({
          tool: "run_block",
          input: { block_id: gmailOutput().block_id, input_data: {} },
          output: gmailOutput(),
        })}
        isLast
      />,
    );

    expectHiddenFromReplay(EMAIL);
  });

  it("keeps a truncated block output out of session replays", () => {
    const preview = `<tool-output-truncated total_chars=90000 workspace_path="tool-outputs/call-1.json">\n${EMAIL}\n</tool-output-truncated>`;
    render(
      <ChainRowView
        row={row({
          tool: "run_capability",
          input: { id: "block:gmail", input: {} },
          output: preview,
        })}
        isLast
      />,
    );

    expectHiddenFromReplay(EMAIL);
  });

  it("leaves other tools' output in session replays", () => {
    render(
      <ChainRowView
        row={row({
          tool: "search_docs",
          text: "Searched docs",
          input: { query: "agents" },
          output: { result: "How to build an agent" },
        })}
        isLast
      />,
    );

    const element = screen.getByText("How to build an agent", { exact: false });
    expect(element.closest(".ph-no-capture")).toBeNull();
    expect(element.closest(".sentry-block")).toBeNull();
  });
});
