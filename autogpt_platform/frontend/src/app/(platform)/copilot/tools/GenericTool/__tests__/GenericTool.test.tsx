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
    const { container } = render(
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
    expect(triggers[0].textContent).toContain(
      'echo "starting simulation run 2"',
    );
    expect(container.textContent).not.toContain("Ran:");
  });

  it("shows 'exit N · <first line of stderr>' on non-zero exit", () => {
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
    expect(trigger.textContent).toContain("missing-bin");
    expect(trigger.textContent).toContain(
      "exit 127 · bash: missing-bin: command not found",
    );
  });

  it("falls back to bare 'exit N' when stderr is empty", () => {
    render(
      <GenericTool
        part={makePart({
          state: "output-available",
          output: { exit_code: 2, stdout: "", stderr: "" },
        })}
      />,
    );
    const trigger = screen.getByRole("button", { expanded: false });
    expect(trigger.textContent).toContain("exit 2");
    expect(trigger.textContent).not.toContain("·");
  });

  it("shows the command and stderr first line for a timed-out command", () => {
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
    expect(trigger.textContent).toContain("sleep 120");
    expect(trigger.textContent).toContain("Timed out after 120s");
  });

  it("uses the command as the row for legacy outputs missing exit_code/timed_out", () => {
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

  it("keeps a quiet single line on exit 0 — stdout only visible after expanding", () => {
    render(
      <GenericTool
        part={makePart({
          state: "output-available",
          input: { command: "cat greeting.txt" },
          output: {
            exit_code: 0,
            stdout: "Hello, world!\nmore lines below\n",
            stderr: "",
          },
        })}
      />,
    );
    const trigger = screen.getByRole("button", { expanded: false });
    expect(trigger.textContent).toContain("cat greeting.txt");
    expect(trigger.textContent).not.toContain("Hello, world!");
    expect(trigger.textContent).not.toContain("completed");

    fireEvent.click(trigger);
    expect(screen.queryByText(/Hello, world!/)).not.toBeNull();
    // Single-section output: no redundant stdout/command labels.
    expect(screen.queryByText("stdout")).toBeNull();
    expect(screen.queryByText("command")).toBeNull();
  });

  it("labels stdout/stderr sections only when both are present", () => {
    render(
      <GenericTool
        part={makePart({
          state: "output-available",
          input: { command: "build.sh" },
          output: {
            exit_code: 1,
            stdout: "compiling...\n",
            stderr: "error: missing semicolon\n",
          },
        })}
      />,
    );
    fireEvent.click(screen.getByRole("button", { expanded: false }));
    expect(screen.queryByText("stdout")).not.toBeNull();
    expect(screen.queryByText("stderr")).not.toBeNull();
  });

  describe("web_search results rendering", () => {
    function makeWebSearchPart(
      results: Array<Record<string, unknown>>,
      query = "kimi k2.6",
      answer = "",
    ): ToolUIPart {
      return {
        type: "tool-web_search",
        toolCallId: "call-web-1",
        state: "output-available",
        input: { query },
        output: {
          type: "web_search_response",
          answer,
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

    it("renders no duplicate 'Searched \u2026' status row once output is available", () => {
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
      // The accordion header already shows count + query; the old
      // MorphingTextAnimation row duplicated the query above it.
      expect(container.textContent).not.toContain("Searched");
      expect(screen.getAllByRole("button").length).toBe(1);
    });

    it("renders the synthesised answer above the citations when present", () => {
      render(
        <GenericTool
          part={makeWebSearchPart(
            [
              { title: "Citation 1", url: "https://example.com/one" },
              { title: "Citation 2", url: "https://example.com/two" },
            ],
            "kimi k2.6 launch",
            "Kimi K2.6 launched on 2026-04-20 with SWE-Bench parity to Opus.",
          )}
        />,
      );
      fireEvent.click(screen.getByRole("button", { expanded: false }));
      expect(
        screen.getByText(/Kimi K2\.6 launched on 2026-04-20/),
      ).not.toBeNull();
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
