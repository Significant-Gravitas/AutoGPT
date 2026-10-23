import { render, screen } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { MessagePart } from "../../ChatMessagesContainer/helpers";
import { ChainRowView } from "../ChainRowView";
import { toChainRow, type ChainRow } from "../helpers";
import { isEmptyResult } from "../resultHelpers";
import { ToolResult } from "../ToolResult";

// What the real tools answer on an account with nothing set up.
const NO_ROUTINES = {
  type: "routines",
  message: "0 routine(s); 0 switched on.",
  session_id: "s-1",
  routines: [],
};
const NO_SCHEDULES = {
  type: "schedule_list",
  message: "No schedules found.",
  session_id: "s-1",
  schedules: [],
};

function done(type: string, input: unknown, output: unknown): MessagePart {
  return {
    type,
    toolCallId: `call-${type}`,
    state: "output-available",
    input,
    output,
  } as MessagePart;
}

function row(output: unknown, tool?: string): ChainRow {
  return {
    key: "r",
    category: "other",
    text: "Tool",
    state: "done",
    output,
    tool,
  };
}

afterEach(cleanup);

describe("empty lookups", () => {
  it("label the row with what came back", () => {
    expect(
      toChainRow(done("tool-list_routines", {}, NO_ROUTINES), 0),
    ).toMatchObject({
      text: "Checked your routines – none yet",
      emptyResult: true,
    });
    expect(
      toChainRow(done("tool-list_schedules", {}, NO_SCHEDULES), 0)?.text,
    ).toBe("Checked your schedules – none yet");
  });

  it("label rows saved before dispatch named the tool that ran", () => {
    // Older chats store the run_capability call itself.
    const legacy = done(
      "tool-run_capability",
      { id: "tool:list_routines", input: {} },
      NO_ROUTINES,
    );
    expect(toChainRow(legacy, 0)?.text).toBe(
      "Checked your routines – none yet",
    );
  });

  it("keep the plain label when something came back", () => {
    const some = { ...NO_ROUTINES, routines: [{ id: "r-1", title: "Digest" }] };
    const plain = toChainRow(done("tool-list_routines", {}, some), 0);
    expect(plain?.text).toBe("Checked your routines");
    expect(plain?.emptyResult).toBeUndefined();
  });

  it("leave nothing to open on a row that already says it", () => {
    const empty = toChainRow(done("tool-list_routines", {}, NO_ROUTINES), 0)!;
    render(<ChainRowView row={empty} isLast />);

    expect(screen.getByText("Checked your routines – none yet")).toBeDefined();
    expect(screen.queryByRole("button")).toBeNull();
    expect(document.body.textContent).not.toContain("[]");
  });
});

describe("the shared result fallback", () => {
  it("reads an empty result as a sentence, never raw brackets", () => {
    render(
      <ToolResult row={row({ widgets: [], filters: {} }, "some_new_tool")} />,
    );

    expect(screen.getByText("Nothing found.")).toBeDefined();
    expect(document.body.textContent).not.toMatch(/\[\]|\{\}/);
  });

  it("never leaves an opened row blank when every field is empty", () => {
    render(<ToolResult row={row({ owner: null }, "some_new_tool")} />);

    expect(screen.getByText("Nothing found.")).toBeDefined();
  });

  it.each([[null], [[]], [{}], ["[]"], [" {} "], ["   "]])(
    "says nothing came back for a bare %j",
    (output) => {
      render(<ToolResult row={row(output, "glob")} />);

      expect(screen.getByText("Nothing found.")).toBeDefined();
      expect(document.body.textContent).not.toMatch(/\[\]|\{\}/);
    },
  );

  it("says so when a card's own payload is empty", () => {
    render(<ToolResult row={row({ result: [] }, "run_mcp_tool")} />);

    expect(screen.getByText("Nothing found.")).toBeDefined();
  });

  it("writes empty values inside a result as None", () => {
    render(
      <ToolResult
        row={row(
          { title: "Digest", tags: [], extra: {}, owner: null },
          "some_new_tool",
        )}
      />,
    );

    expect(screen.getByText("Digest")).toBeDefined();
    expect(screen.getAllByText("None")).toHaveLength(2);
    expect(screen.queryByText("Owner")).toBeNull();
    expect(document.body.textContent).not.toMatch(/\[\]|\{\}|null/);
  });
});

describe("isEmptyResult", () => {
  it.each([
    [NO_ROUTINES, true],
    [NO_SCHEDULES, true],
    [{ ...NO_ROUTINES, expert_id: "e-1" }, true],
    [{ presets: [], total_count: 0, page: 1, page_size: 20 }, true],
    // A page past the end: the collection is not empty.
    [{ presets: [], total_count: 5, page: 2, page_size: 20 }, false],
    [{ routines: [{ id: "r-1" }] }, false],
    [{ name: "scraper", triggers: [] }, false],
    [{ ok: true, failed: [] }, false],
    [{ type: "done", message: "Deleted." }, false],
    [[], false],
    ["", false],
    [undefined, false],
  ])("%j → %s", (output, expected) => {
    expect(isEmptyResult(output)).toBe(expected);
  });
});
