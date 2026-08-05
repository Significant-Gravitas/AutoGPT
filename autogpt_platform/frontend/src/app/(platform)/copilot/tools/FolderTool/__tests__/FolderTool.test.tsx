import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { ToolUIPart } from "ai";
import { FolderTool } from "../FolderTool";

afterEach(cleanup);

const FOLDER = {
  id: "f1",
  name: "Research",
  agent_count: 2,
  subfolder_count: 0,
};

function makePart(output: unknown, state = "output-available") {
  return {
    type: "tool-folder_tool",
    toolCallId: "call-folder",
    state,
    input: {},
    output,
  } as unknown as ToolUIPart;
}

// The status and accordion glyphs are picked by branching on the output type.
// A wrong or undefined icon import renders an empty <svg> instead of throwing,
// so compare the drawn geometry rather than asserting the element exists.
function glyphs(root: HTMLElement) {
  return Array.from(root.querySelectorAll("svg"))
    .map((svg) =>
      Array.from(svg.querySelectorAll("path,circle,rect,line,polyline"))
        .map((n) => n.getAttribute("d") ?? n.outerHTML)
        .join("|"),
    )
    .filter(Boolean);
}

function renderTool(output: unknown, state?: string) {
  return render(<FolderTool part={makePart(output, state)} />);
}

function statusGlyph(output: unknown, state?: string) {
  const { container, unmount } = renderTool(output, state);
  const g = glyphs(container)[0];
  unmount();
  return g;
}

describe("FolderTool status icon", () => {
  it("shows the folder glyph once the call has settled", () => {
    const glyph = statusGlyph({
      type: "folder_created",
      message: "Created",
      folder: FOLDER,
    });
    expect(glyph).toBeTruthy();
  });

  it("swaps in the alert glyph when the part itself errored", () => {
    const ok = statusGlyph({
      type: "folder_created",
      message: "Created",
      folder: FOLDER,
    });
    const failed = statusGlyph(
      { type: "folder_created", message: "Created", folder: FOLDER },
      "output-error",
    );

    expect(failed).toBeTruthy();
    expect(failed).not.toBe(ok);
  });

  it("treats an error-shaped output as an error even when the part succeeded", () => {
    const failed = statusGlyph({ type: "error", message: "nope" });
    const errored = statusGlyph(
      { type: "folder_created", message: "Created", folder: FOLDER },
      "output-error",
    );

    expect(failed).toBe(errored);
  });

  it("renders no accordion for an error output", () => {
    renderTool({ type: "error", message: "nope" });
    expect(screen.queryByRole("button")).toBeNull();
  });
});

describe("FolderTool accordion icon", () => {
  function accordionGlyph(output: unknown) {
    const { container, unmount } = renderTool(output);
    const drawn = glyphs(container);
    // [0] is the status row; the accordion header glyph follows it.
    expect(drawn.length).toBeGreaterThanOrEqual(2);
    const g = drawn[1];
    unmount();
    return g;
  }

  it("gives create, list, delete and move their own glyphs", () => {
    const created = accordionGlyph({
      type: "folder_created",
      message: "Created",
      folder: FOLDER,
    });
    const listed = accordionGlyph({
      type: "folder_list",
      message: "Listed",
      folders: [FOLDER],
      count: 1,
    });
    const deleted = accordionGlyph({
      type: "folder_deleted",
      message: "Deleted",
      folder_id: "f1",
    });
    const moved = accordionGlyph({
      type: "folder_moved",
      message: "Moved",
      folder: FOLDER,
    });

    const all = [created, listed, deleted, moved];
    all.forEach((g) => expect(g).toBeTruthy());
    expect(new Set(all).size).toBe(all.length);
  });

  it("falls back to the plain folder glyph for updates", () => {
    const updated = accordionGlyph({
      type: "folder_updated",
      message: "Renamed",
      folder: FOLDER,
    });
    const moved = accordionGlyph({
      type: "folder_moved",
      message: "Moved",
      folder: FOLDER,
    });

    expect(updated).toBe(moved);
  });
});

describe("FolderTool tree icons", () => {
  it("uses different glyphs for expanded and collapsed folders", () => {
    renderTool({
      type: "folder_list",
      message: "Listed",
      folders: [],
      tree: [
        {
          ...FOLDER,
          children: [],
          agents: [{ id: "a1", name: "Agent" }],
        },
      ],
      count: 1,
    });
    const folderButton = screen.getByRole("button", {
      name: /Research \(2 agents\)/,
    });
    const expandedGlyph = glyphs(folderButton)[0];

    fireEvent.click(folderButton);

    const collapsedGlyph = glyphs(folderButton)[0];
    expect(expandedGlyph).toBeTruthy();
    expect(collapsedGlyph).toBeTruthy();
    expect(collapsedGlyph).not.toBe(expandedGlyph);
  });
});
