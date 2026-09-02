import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { ChainRow } from "../helpers";
import {
  FileCard,
  KeyValueList,
  OutputList,
  SearchResults,
  Terminal,
  TodoList,
} from "../ToolResultViews";

function row(output: unknown, input?: unknown): ChainRow {
  return {
    key: "tool",
    category: "other",
    text: "Tool",
    state: "done",
    output,
    input,
  };
}

describe("SearchResults", () => {
  afterEach(cleanup);

  it("renders linked results with their domain and an answer", () => {
    render(
      <SearchResults
        items={[{ title: "Wiki page", url: "https://en.wikipedia.org/wiki/X" }]}
        answer="Paris is the capital"
      />,
    );

    expect(screen.getByText("Paris is the capital")).toBeDefined();
    const link = screen.getByRole("link", { name: "Wiki page" });
    expect(link.getAttribute("href")).toBe("https://en.wikipedia.org/wiki/X");
    expect(screen.getByText("en.wikipedia.org")).toBeDefined();
  });

  it("renders unlinked results as plain text", () => {
    render(<SearchResults items={[{ snippet: "Loose snippet" }]} />);

    expect(screen.getByText("Loose snippet")).toBeDefined();
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("falls back to inline JSON for untitled results", () => {
    render(<SearchResults items={[{ score: 3 }]} />);

    expect(screen.getByText('{"score":3}')).toBeDefined();
  });

  it("swaps the favicon for a globe icon when it fails to load", () => {
    const { container } = render(
      <SearchResults
        items={[{ title: "Wiki page", url: "https://en.wikipedia.org/wiki/X" }]}
      />,
    );

    const favicon = container.querySelector("img");
    expect(favicon).not.toBeNull();

    fireEvent.error(favicon as HTMLImageElement);

    expect(container.querySelector("img")).toBeNull();
  });
});

describe("Terminal", () => {
  afterEach(cleanup);

  it("shows the command, stdout and a non-zero exit code", () => {
    render(
      <Terminal
        row={row({ stdout: "file.txt", exit_code: 2 }, { command: "ls" })}
      />,
    );

    expect(screen.getByText("ls")).toBeDefined();
    expect(screen.getByText("file.txt")).toBeDefined();
    expect(screen.getByText("exit 2")).toBeDefined();
  });

  it("hides the exit line for successful commands", () => {
    render(
      <Terminal
        row={row({ stdout: "ok", exit_code: 0 }, { command: "true" })}
      />,
    );

    expect(screen.getByText("ok")).toBeDefined();
    expect(screen.queryByText("exit 0")).toBeNull();
  });

  it("uses stderr when stdout is missing", () => {
    render(<Terminal row={row({ stderr: "boom" }, { command: "make" })} />);

    expect(screen.getByText("boom")).toBeDefined();
  });
});

describe("TodoList", () => {
  afterEach(cleanup);

  it("renders todos across statuses", () => {
    render(
      <TodoList
        row={row(
          { ok: true },
          {
            todos: [
              { content: "Task A", status: "completed" },
              { content: "Task B", status: "in_progress" },
              { activeForm: "Doing Task C", status: "pending" },
            ],
          },
        )}
      />,
    );

    expect(screen.getByText("Task A")).toBeDefined();
    expect(screen.getByText("Task B")).toBeDefined();
    expect(screen.getByText("Doing Task C")).toBeDefined();
  });

  it("falls back to the raw output when no todos exist", () => {
    render(<TodoList row={row({ note: "no todos" }, {})} />);

    expect(screen.getByText("Note")).toBeDefined();
    expect(screen.getByText("no todos")).toBeDefined();
  });
});

describe("FileCard", () => {
  afterEach(cleanup);

  it("shows the path, size and preview from the row", () => {
    render(
      <FileCard
        row={row(
          {
            size_bytes: 2048,
            mime_type: "image/png",
            preview: "binary preview",
          },
          { file_path: "chart.png" },
        )}
      />,
    );

    expect(screen.getByText("chart.png")).toBeDefined();
    expect(screen.getByText("2.0 KB")).toBeDefined();
    expect(screen.getByText("binary preview")).toBeDefined();
  });

  it("reads the path from the output when the input lacks one", () => {
    render(<FileCard row={row({ path: "out.txt", size: 12 })} />);

    expect(screen.getByText("out.txt")).toBeDefined();
    expect(screen.getByText("12 B")).toBeDefined();
  });

  it("falls back to key/value output without any path", () => {
    render(<FileCard row={row({ status: "deleted" })} />);

    expect(screen.getByText("Status")).toBeDefined();
    expect(screen.getByText("deleted")).toBeDefined();
  });
});

describe("OutputList", () => {
  afterEach(cleanup);

  it("labels items by name and prints values", () => {
    render(
      <OutputList
        items={[
          { name: "summary", value: "All good" },
          { value: "unnamed value" },
        ]}
      />,
    );

    expect(screen.getByText("summary")).toBeDefined();
    expect(screen.getByText("All good")).toBeDefined();
    expect(screen.getByText("Output 2")).toBeDefined();
    expect(screen.getByText("unnamed value")).toBeDefined();
  });
});

describe("KeyValueList", () => {
  afterEach(cleanup);

  it("renders humanized keys with inline values", () => {
    render(<KeyValueList value={{ status_code: 200, ok: true }} />);

    expect(screen.getByText("Status code")).toBeDefined();
    expect(screen.getByText("200")).toBeDefined();
    expect(screen.getByText("Ok")).toBeDefined();
    expect(screen.getByText("true")).toBeDefined();
  });

  it("renders plain strings as preformatted text", () => {
    render(<KeyValueList value="raw output" />);

    expect(screen.getByText("raw output")).toBeDefined();
  });

  it("renders nothing for empty strings", () => {
    const { container } = render(<KeyValueList value="   " />);

    expect(container.firstChild).toBeNull();
  });

  it("renders nothing for empty objects", () => {
    const { container } = render(<KeyValueList value={{}} />);

    expect(container.firstChild).toBeNull();
  });
});
