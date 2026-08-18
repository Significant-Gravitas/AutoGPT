import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { ToolUIPart } from "ai";
import { CollapsedToolGroup } from "../CollapsedToolGroup";

afterEach(cleanup);

function makePart(toolName: string, overrides: Partial<ToolUIPart> = {}) {
  return {
    type: `tool-${toolName}`,
    toolCallId: `call-${toolName}`,
    state: "output-available",
    input: {},
    output: { message: "ok" },
    ...overrides,
  } as unknown as ToolUIPart;
}

// The entry glyph is chosen by a switch on the tool category. A wrong or
// undefined icon import renders an empty <svg> instead of throwing, so assert
// on the drawn geometry rather than on the element merely existing.
function glyphs(root: HTMLElement) {
  return Array.from(root.querySelectorAll("svg"))
    .map((svg) =>
      Array.from(svg.querySelectorAll("path,circle,rect,line,polyline"))
        .map((n) => n.getAttribute("d") ?? n.outerHTML)
        .join("|"),
    )
    .filter(Boolean);
}

describe("CollapsedToolGroup", () => {
  it("summarises completed calls and stays collapsed until clicked", () => {
    render(
      <CollapsedToolGroup parts={[makePart("bash_exec"), makePart("grep")]} />,
    );

    const toggle = screen.getByRole("button");
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(toggle.textContent).toContain("2 tool calls completed");
    expect(screen.queryByText(/Ran command/i)).toBeNull();
  });

  it("counts failures in the summary label", () => {
    render(
      <CollapsedToolGroup
        parts={[
          makePart("bash_exec"),
          makePart("grep", { state: "output-error" } as Partial<ToolUIPart>),
        ]}
      />,
    );

    expect(screen.getByRole("button").textContent).toContain(
      "2 tool calls (1 failed)",
    );
  });

  it("reveals one entry per tool call when expanded", () => {
    const { container } = render(
      <CollapsedToolGroup
        parts={[makePart("bash_exec"), makePart("grep"), makePart("glob")]}
      />,
    );

    const toggle = screen.getByRole("button");
    fireEvent.click(toggle);

    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    const panel = container.querySelector(
      `#${CSS.escape(toggle.getAttribute("aria-controls") ?? "")}`,
    );
    expect(panel).not.toBeNull();
    expect(panel?.children.length).toBe(3);
  });

  it("draws a distinct glyph for every tool category", () => {
    const byCategory: [string, string][] = [
      ["bash", "bash_exec"],
      ["web", "web_search"],
      ["browser", "browser_navigate"],
      ["file-read", "read_file"],
      ["file-delete", "delete_workspace_file"],
      ["file-list", "glob"],
      ["search", "grep"],
      ["edit", "edit_file"],
      ["todo", "TodoWrite"],
      ["compaction", "context_compaction"],
      ["other", "some_unknown_tool"],
    ];

    const seen: string[] = [];
    for (const [, toolName] of byCategory) {
      const { container, unmount } = render(
        <CollapsedToolGroup parts={[makePart(toolName)]} />,
      );
      fireEvent.click(screen.getByRole("button"));
      // [0] and [1] belong to the toggle (caret + status); the entry is last.
      const drawn = glyphs(container);
      expect(drawn.length).toBe(3);
      seen.push(drawn[2]);
      unmount();
    }

    expect(new Set(seen).size).toBe(seen.length);
  });

  it("swaps the entry glyph for the alert icon when a call errored", () => {
    const { container: ok } = render(
      <CollapsedToolGroup parts={[makePart("bash_exec")]} />,
    );
    fireEvent.click(screen.getByRole("button"));
    const okGlyph = glyphs(ok)[2];
    cleanup();

    const { container: failed } = render(
      <CollapsedToolGroup
        parts={[
          makePart("bash_exec", {
            state: "output-error",
          } as Partial<ToolUIPart>),
        ]}
      />,
    );
    fireEvent.click(screen.getByRole("button"));
    const failedGlyph = glyphs(failed)[2];

    expect(failedGlyph).toBeTruthy();
    expect(failedGlyph).not.toBe(okGlyph);
  });

  it("shares the file glyph between reads and writes", () => {
    const { container: read } = render(
      <CollapsedToolGroup parts={[makePart("read_file")]} />,
    );
    fireEvent.click(screen.getByRole("button"));
    const readGlyph = glyphs(read)[2];
    cleanup();

    const { container: write } = render(
      <CollapsedToolGroup parts={[makePart("write_file")]} />,
    );
    fireEvent.click(screen.getByRole("button"));

    expect(glyphs(write)[2]).toBe(readGlyph);
  });
});
