import { describe, expect, it } from "vitest";
import type { ToolUIPart } from "ai";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { GenericTool } from "../GenericTool";

function makePart(overrides: Record<string, unknown> = {}): ToolUIPart {
  return {
    type: "tool-bash_exec",
    toolCallId: "call-1",
    state: "input-streaming",
    input: { command: 'echo "hi"' },
    ...overrides,
  } as ToolUIPart;
}

describe("GenericTool", () => {
  it("shows a subtitle and no accordion while the tool is streaming", () => {
    const { container } = render(
      <GenericTool part={makePart({ state: "input-streaming" })} />,
    );
    expect(screen.queryByRole("button")).toBeNull();
    expect(container.textContent).toContain("Running");
  });

  it("renders exactly one row once output is available (accordion only, no loose status line)", () => {
    render(
      <GenericTool
        part={makePart({
          state: "output-available",
          input: { command: 'echo "starting simulation run 2"' },
          output: { exit_code: 1, stdout: "", stderr: "boom" },
        })}
      />,
    );
    // The accordion trigger is the only interactive element; no separate
    // MorphingTextAnimation status row is rendered alongside it.
    const triggers = screen.getAllByRole("button");
    expect(triggers.length).toBe(1);
    expect(triggers[0].textContent).toContain("Command failed (exit 1)");
  });

  it("shows 'status code N · <first line of stderr>' on non-zero exit", () => {
    render(
      <GenericTool
        part={makePart({
          state: "output-available",
          input: { command: "missing-bin" },
          output: {
            exit_code: 127,
            stdout: "",
            stderr: "bash: missing-bin: command not found\n",
          },
        })}
      />,
    );
    const trigger = screen.getByRole("button", { expanded: false });
    expect(trigger.textContent).toContain("Command failed (exit 127)");
    expect(trigger.textContent).toContain(
      "status code 127 · bash: missing-bin: command not found",
    );
  });

  it("falls back to bare 'status code N' when stderr is empty", () => {
    render(
      <GenericTool
        part={makePart({
          state: "output-available",
          output: { exit_code: 2, stdout: "", stderr: "" },
        })}
      />,
    );
    const trigger = screen.getByRole("button", { expanded: false });
    expect(trigger.textContent).toContain("status code 2");
    expect(trigger.textContent).not.toContain("·");
  });

  it("shows the stderr first line for a timed-out command", () => {
    render(
      <GenericTool
        part={makePart({
          state: "output-available",
          input: { command: "sleep 120" },
          output: {
            exit_code: -1,
            timed_out: true,
            stderr: "Timed out after 120s",
          },
        })}
      />,
    );
    const trigger = screen.getByRole("button", { expanded: false });
    expect(trigger.textContent).toContain("Command timed out");
    expect(trigger.textContent).toContain("Timed out after 120s");
    expect(trigger.textContent).not.toContain("sleep 120");
  });

  it("falls back to the command preview for legacy outputs missing exit_code/timed_out", () => {
    render(
      <GenericTool
        part={makePart({
          state: "output-available",
          input: { command: "echo hello" },
          output: { stdout: "hello\n" },
        })}
      />,
    );
    const trigger = screen.getByRole("button", { expanded: false });
    expect(trigger.textContent).toContain("echo hello");
  });

  it("prefers stdout first line on exit 0, falls back to 'completed'", () => {
    const { rerender } = render(
      <GenericTool
        part={makePart({
          state: "output-available",
          output: {
            exit_code: 0,
            stdout: "Hello, world!\nmore lines below\n",
            stderr: "",
          },
        })}
      />,
    );
    const trigger1 = screen.getByRole("button", { expanded: false });
    expect(trigger1.textContent).toContain("Hello, world!");
    expect(trigger1.textContent).not.toContain("more lines below");

    rerender(
      <GenericTool
        part={makePart({
          state: "output-available",
          output: { exit_code: 0, stdout: "", stderr: "" },
        })}
      />,
    );
    const trigger2 = screen.getByRole("button", { expanded: false });
    expect(trigger2.textContent).toContain("completed");
  });

  describe("web_search results rendering", () => {
    function makeWebSearchPart(
      results: Array<Record<string, unknown>>,
      query = "kimi k2.6",
    ): ToolUIPart {
      return {
        type: "tool-web_search",
        toolCallId: "call-web-1",
        state: "output-available",
        input: { query },
        output: {
          type: "web_search_response",
          results,
          query,
          search_requests: 1,
        },
      } as unknown as ToolUIPart;
    }

    it("renders an 'N search results' title and shows the query in the description", () => {
      render(
        <GenericTool
          part={makeWebSearchPart([
            {
              title: "Kimi K2.6 release notes",
              url: "https://example.com/kimi",
              snippet: "A fast model",
              page_age: "2 days ago",
            },
            {
              title: "Second result",
              url: "https://example.com/two",
              snippet: "Another snippet",
            },
          ])}
        />,
      );
      const trigger = screen.getByRole("button", { expanded: false });
      expect(trigger.textContent).toContain("2 search results");
      expect(trigger.textContent).toContain("kimi k2.6");

      fireEvent.click(trigger);

      const firstLink = screen.getByRole("link", {
        name: "Kimi K2.6 release notes",
      }) as HTMLAnchorElement;
      expect(firstLink.getAttribute("href")).toBe("https://example.com/kimi");
      expect(firstLink.getAttribute("target")).toBe("_blank");
      expect(firstLink.getAttribute("rel")).toBe("noopener noreferrer");
      expect(screen.queryByText("A fast model")).not.toBeNull();
      expect(screen.queryByText("2 days ago")).not.toBeNull();

      const secondLink = screen.getByRole("link", {
        name: "Second result",
      }) as HTMLAnchorElement;
      expect(secondLink.getAttribute("href")).toBe("https://example.com/two");
    });

    it("uses singular 'search result' when there is exactly one result", () => {
      render(
        <GenericTool
          part={makeWebSearchPart([
            {
              title: "Only result",
              url: "https://example.com/only",
              snippet: "Lone snippet",
            },
          ])}
        />,
      );
      const trigger = screen.getByRole("button", { expanded: false });
      expect(trigger.textContent).toContain("1 search result");
      expect(trigger.textContent).not.toContain("1 search results");
    });

    it("handles an empty results array (0 search results)", () => {
      render(<GenericTool part={makeWebSearchPart([])} />);
      const trigger = screen.getByRole("button", { expanded: false });
      expect(trigger.textContent).toContain("0 search results");
    });

    it("renders an untitled non-link when a result has no url", () => {
      render(
        <GenericTool
          part={makeWebSearchPart([
            { title: "No URL entry", snippet: "Just text" },
          ])}
        />,
      );
      fireEvent.click(screen.getByRole("button", { expanded: false }));
      expect(screen.queryByRole("link")).toBeNull();
      expect(screen.queryByText("No URL entry")).not.toBeNull();
      expect(screen.queryByText("Just text")).not.toBeNull();
    });

    it("shows subtitle 'Searched \"…\"' once web_search output is available", () => {
      const { container } = render(
        <GenericTool
          part={makeWebSearchPart(
            [
              {
                title: "Kimi K2.6 release notes",
                url: "https://example.com/kimi",
                snippet: "A fast model",
              },
            ],
            "kimi k2.6",
          )}
        />,
      );
      // MorphingTextAnimation splits each character into its own span and
      // substitutes spaces with  , so assert on a normalized textContent
      // rather than the raw substring.
      const normalized = (container.textContent ?? "").replace(/ /g, " ");
      expect(normalized).toContain('Searched "kimi k2.6"');
    });

    it("uses '(untitled)' when a search result has no title", () => {
      render(
        <GenericTool
          part={makeWebSearchPart([
            { url: "https://example.com/x", snippet: "No title here" },
          ])}
        />,
      );
      fireEvent.click(screen.getByRole("button", { expanded: false }));
      const link = screen.getByRole("link", {
        name: "(untitled)",
      }) as HTMLAnchorElement;
      expect(link.getAttribute("href")).toBe("https://example.com/x");
    });
  });

  describe("getWebAccordionData non-results fallback", () => {
    function makeWebFetchPart(output: Record<string, unknown>): ToolUIPart {
      return {
        type: "tool-web_fetch",
        toolCallId: "call-fetch-1",
        state: "output-available",
        input: { url: "https://example.com/page" },
        output,
      } as unknown as ToolUIPart;
    }

    it("renders 'Web fetch' title when output has content instead of results", () => {
      render(
        <GenericTool part={makeWebFetchPart({ content: "fetched body" })} />,
      );
      const trigger = screen.getByRole("button", { expanded: false });
      expect(trigger.textContent).toContain("Web fetch");
      fireEvent.click(trigger);
      expect(screen.queryByText("fetched body")).not.toBeNull();
    });

    it("renders 'Response (N)' title when output has a status_code", () => {
      render(
        <GenericTool
          part={makeWebFetchPart({ status_code: 404, message: "not found" })}
        />,
      );
      const trigger = screen.getByRole("button", { expanded: false });
      expect(trigger.textContent).toContain("Response (404)");
    });

    it("falls back to MCP text blocks when direct content is absent", () => {
      render(
        <GenericTool
          part={makeWebFetchPart({
            content: [{ type: "text", text: "mcp body" }],
          })}
        />,
      );
      fireEvent.click(screen.getByRole("button", { expanded: false }));
      expect(screen.queryByText("mcp body")).not.toBeNull();
    });
  });
});
