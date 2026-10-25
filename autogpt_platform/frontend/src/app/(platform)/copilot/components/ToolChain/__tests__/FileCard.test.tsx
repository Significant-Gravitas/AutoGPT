import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { FileCard } from "../FileCard";
import type { ChainRow } from "../helpers";

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

describe("FileCard", () => {
  afterEach(cleanup);

  it("shows the file's name, size and preview, with the path behind details", () => {
    render(
      <FileCard
        row={row(
          {
            size_bytes: 2048,
            mime_type: "image/png",
            preview: "binary preview",
          },
          { file_path: "~/workspace/dewleaf/sales_chart.png" },
        )}
      />,
    );

    expect(screen.getByText("Sales Chart")).toBeDefined();
    expect(screen.getByText("2.0 KB")).toBeDefined();
    expect(screen.getByText("binary preview")).toBeDefined();
    expect(screen.queryByText(/workspace|\.png/)).toBeNull();

    fireEvent.click(
      screen.getByRole("button", { name: "Show technical details" }),
    );
    expect(
      screen.getByText("~/workspace/dewleaf/sales_chart.png"),
    ).toBeDefined();
  });

  it("reads the path from the output when the input lacks one", () => {
    render(<FileCard row={row({ path: "out.txt", size: 12 })} />);

    expect(screen.getByText("Out")).toBeDefined();
    expect(screen.getByText("12 B")).toBeDefined();
  });

  it("calls a file whose name still reads as code just a file", () => {
    render(
      <FileCard row={row({ size: 12 }, { path: "./build/app.min.js" })} />,
    );

    expect(screen.getByText("File")).toBeDefined();
    expect(screen.queryByText(/app\.min/)).toBeNull();
  });

  it("falls back to key/value output without any path", () => {
    render(<FileCard row={row({ status: "deleted" })} />);

    expect(screen.getByText("Status")).toBeDefined();
    expect(screen.getByText("deleted")).toBeDefined();
  });
});
