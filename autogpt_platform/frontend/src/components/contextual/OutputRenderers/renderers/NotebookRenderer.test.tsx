import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type React from "react";
import { renderToString } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { notebookRenderer } from "./NotebookRenderer";

// ---------------------------------------------------------------------------
// Helper: find text content inside any <pre> element
// ---------------------------------------------------------------------------

function findPreWithText(
  container: HTMLElement,
  text: string,
): HTMLPreElement | undefined {
  return Array.from(container.querySelectorAll("pre")).find((pre) =>
    pre.textContent?.includes(text),
  );
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const minimalNotebook = {
  nbformat: 4,
  nbformat_minor: 5,
  metadata: {
    kernelspec: {
      language: "python",
      display_name: "Python 3",
      name: "python3",
    },
    language_info: { name: "python", version: "3.10.0" },
  },
  cells: [
    {
      cell_type: "code" as const,
      source: "print('hello')",
      execution_count: 1,
      metadata: {},
      outputs: [
        {
          output_type: "stream" as const,
          name: "stdout" as const,
          text: "hello\n",
        },
      ],
    },
  ],
};

const multiCellNotebook = {
  nbformat: 4,
  nbformat_minor: 5,
  metadata: {
    kernelspec: { language: "python" },
    language_info: { name: "python", version: "3.11.0" },
  },
  cells: [
    {
      cell_type: "markdown" as const,
      source: "# Hello World",
      metadata: {},
    },
    {
      cell_type: "code" as const,
      source: ["x = 1\n", "y = 2\n", "print(x + y)"],
      execution_count: 1,
      metadata: {},
      outputs: [
        {
          output_type: "stream" as const,
          name: "stdout" as const,
          text: ["3\n"],
        },
      ],
    },
    {
      cell_type: "code" as const,
      source: "raise ValueError('oops')",
      execution_count: 2,
      metadata: {},
      outputs: [
        {
          output_type: "error" as const,
          ename: "ValueError",
          evalue: "oops",
          traceback: [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mValueError\u001b[0m: oops",
          ],
        },
      ],
    },
    {
      cell_type: "raw" as const,
      source: "raw content here",
      metadata: {},
    },
    {
      cell_type: "raw" as const,
      source: "   ",
      metadata: {},
    },
    {
      cell_type: "code" as const,
      source: "1 + 1",
      execution_count: null,
      metadata: {},
      outputs: [],
    },
  ],
};

const displayDataNotebook = {
  nbformat: 4,
  cells: [
    {
      cell_type: "code" as const,
      source: "display(img)",
      execution_count: 1,
      outputs: [
        {
          output_type: "display_data" as const,
          data: {
            "image/png": "iVBORw0KGgo=",
            "text/plain": "<Figure>",
          },
        },
      ],
    },
  ],
};

const jpegOutputNotebook = {
  nbformat: 4,
  cells: [
    {
      cell_type: "code" as const,
      source: "show(jpeg)",
      execution_count: 1,
      outputs: [
        {
          output_type: "display_data" as const,
          data: {
            "image/jpeg": "/9j/4AAQSkZJRg==",
          },
        },
      ],
    },
  ],
};

const svgOutputNotebook = {
  nbformat: 4,
  cells: [
    {
      cell_type: "code" as const,
      source: "show(svg)",
      execution_count: 1,
      outputs: [
        {
          output_type: "display_data" as const,
          data: {
            "image/svg+xml":
              '<svg xmlns="http://www.w3.org/2000/svg"><circle r="10"/></svg>',
          },
        },
      ],
    },
  ],
};

const htmlOutputNotebook = {
  nbformat: 4,
  cells: [
    {
      cell_type: "code" as const,
      source: "display(html)",
      execution_count: 1,
      outputs: [
        {
          output_type: "display_data" as const,
          data: {
            "text/html": "<table><tr><td>cell</td></tr></table>",
          },
        },
      ],
    },
  ],
};

const executeResultNotebook = {
  nbformat: 4,
  cells: [
    {
      cell_type: "code" as const,
      source: "42",
      execution_count: 3,
      outputs: [
        {
          output_type: "execute_result" as const,
          execution_count: 3,
          data: {
            "text/plain": "42",
          },
        },
      ],
    },
  ],
};

const stderrNotebook = {
  nbformat: 4,
  cells: [
    {
      cell_type: "code" as const,
      source: "import warnings; warnings.warn('test')",
      execution_count: 1,
      outputs: [
        {
          output_type: "stream" as const,
          name: "stderr" as const,
          text: "UserWarning: test\n",
        },
      ],
    },
  ],
};

const noMetadataNotebook = {
  nbformat: 4,
  cells: [
    {
      cell_type: "code" as const,
      source: "1+1",
      execution_count: 1,
      outputs: [],
    },
  ],
};

const kernelOnlyNotebook = {
  nbformat: 4,
  metadata: {
    kernelspec: { language: "R" },
  },
  cells: [
    {
      cell_type: "code" as const,
      source: "print('hi')",
      execution_count: 1,
      outputs: [],
    },
  ],
};

const arraySourceNotebook = {
  nbformat: 4,
  cells: [
    {
      cell_type: "code" as const,
      source: ["line1\n", "line2\n"],
      execution_count: 1,
      outputs: [
        {
          output_type: "stream" as const,
          name: "stdout" as const,
          text: ["out1\n", "out2\n"],
        },
      ],
    },
  ],
};

const unknownOutputNotebook = {
  nbformat: 4,
  cells: [
    {
      cell_type: "code" as const,
      source: "foo()",
      execution_count: 1,
      outputs: [
        {
          output_type: "unknown_type" as any,
          data: { "text/plain": "should not render" },
        },
      ],
    },
  ],
};

const emptyStreamNotebook = {
  nbformat: 4,
  cells: [
    {
      cell_type: "code" as const,
      source: "pass",
      execution_count: 1,
      outputs: [
        {
          output_type: "stream" as const,
          name: "stdout" as const,
          text: "",
        },
      ],
    },
  ],
};

const displayDataNoMatchNotebook = {
  nbformat: 4,
  cells: [
    {
      cell_type: "code" as const,
      source: "custom()",
      execution_count: 1,
      outputs: [
        {
          output_type: "display_data" as const,
          data: {},
        },
      ],
    },
  ],
};

// ---------------------------------------------------------------------------
// canRender
// ---------------------------------------------------------------------------

describe("notebookRenderer.canRender", () => {
  it("matches metadata type 'notebook' for valid notebook content", () => {
    expect(
      notebookRenderer.canRender(minimalNotebook, { type: "notebook" }),
    ).toBe(true);
  });

  it("matches .ipynb filename with valid notebook content", () => {
    expect(
      notebookRenderer.canRender(minimalNotebook, {
        filename: "analysis.ipynb",
      }),
    ).toBe(true);
    expect(
      notebookRenderer.canRender(minimalNotebook, { filename: "REPORT.IPYNB" }),
    ).toBe(true);
    expect(
      notebookRenderer.canRender(minimalNotebook, {
        filename: "data.Ipynb",
      }),
    ).toBe(true);
  });

  it("matches application/x-ipynb+json mimeType with valid notebook content", () => {
    expect(
      notebookRenderer.canRender(minimalNotebook, {
        mimeType: "application/x-ipynb+json",
      }),
    ).toBe(true);
  });

  it("rejects notebook metadata when content is invalid", () => {
    expect(notebookRenderer.canRender("anything", { type: "notebook" })).toBe(
      false,
    );
    expect(
      notebookRenderer.canRender("anything", { filename: "analysis.ipynb" }),
    ).toBe(false);
    expect(
      notebookRenderer.canRender("anything", {
        mimeType: "application/x-ipynb+json",
      }),
    ).toBe(false);
  });

  it("detects valid notebook object", () => {
    expect(notebookRenderer.canRender(minimalNotebook)).toBe(true);
  });

  it("detects valid notebook JSON string", () => {
    expect(notebookRenderer.canRender(JSON.stringify(minimalNotebook))).toBe(
      true,
    );
  });

  it("rejects object without nbformat", () => {
    expect(notebookRenderer.canRender({ cells: [] })).toBe(false);
  });

  it("rejects object without cells", () => {
    expect(notebookRenderer.canRender({ nbformat: 4 })).toBe(false);
  });

  it("rejects object with non-array cells", () => {
    expect(
      notebookRenderer.canRender({ nbformat: 4, cells: "not array" }),
    ).toBe(false);
  });

  it("rejects notebook-shaped objects with invalid cells", () => {
    expect(
      notebookRenderer.canRender({
        nbformat: 4,
        cells: [{ source: "missing type" }],
      }),
    ).toBe(false);
    expect(
      notebookRenderer.canRender({
        nbformat: 4,
        cells: [{ cell_type: "code" }],
      }),
    ).toBe(false);
  });

  it("rejects invalid JSON string", () => {
    expect(notebookRenderer.canRender("{not valid json}")).toBe(false);
  });

  it("rejects plain string", () => {
    expect(notebookRenderer.canRender("just some text")).toBe(false);
  });

  it("rejects null and undefined", () => {
    expect(notebookRenderer.canRender(null)).toBe(false);
    expect(notebookRenderer.canRender(undefined)).toBe(false);
  });

  it("rejects number values", () => {
    expect(notebookRenderer.canRender(42)).toBe(false);
  });

  it("rejects non-notebook objects", () => {
    expect(notebookRenderer.canRender({ key: "value" })).toBe(false);
    expect(notebookRenderer.canRender([1, 2, 3])).toBe(false);
  });

  it("rejects without metadata when value is not a notebook", () => {
    expect(notebookRenderer.canRender("random", {})).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------

describe("notebookRenderer.render", () => {
  it("returns null for invalid notebook", () => {
    expect(notebookRenderer.render("not a notebook")).toBeNull();
    expect(notebookRenderer.render(null)).toBeNull();
    expect(notebookRenderer.render(42)).toBeNull();
  });

  it("renders notebook header with language, version, nbformat, and cell count", () => {
    render(notebookRenderer.render(minimalNotebook) as React.ReactElement);
    expect(screen.getByText(/python 3\.10\.0/i)).toBeDefined();
    expect(screen.getByText("nbformat 4")).toBeDefined();
    expect(screen.getByText("1 cells")).toBeDefined();
  });

  it("renders code cell with execution count gutter", () => {
    render(notebookRenderer.render(minimalNotebook) as React.ReactElement);
    expect(screen.getByText("[1]:")).toBeDefined();
    expect(screen.getByText("print('hello')")).toBeDefined();
  });

  it("renders code cell with null execution count as [ ]:", () => {
    render(notebookRenderer.render(multiCellNotebook) as React.ReactElement);
    expect(screen.getByText("[ ]:")).toBeDefined();
  });

  it("renders markdown cells", () => {
    render(notebookRenderer.render(multiCellNotebook) as React.ReactElement);
    expect(screen.getByText("Hello World")).toBeDefined();
  });

  it("renders raw cells with content", () => {
    render(notebookRenderer.render(multiCellNotebook) as React.ReactElement);
    expect(screen.getByText("raw content here")).toBeDefined();
  });

  it("skips empty raw cells", () => {
    const { container } = render(
      notebookRenderer.render(multiCellNotebook) as React.ReactElement,
    );
    const pres = container.querySelectorAll("pre");
    const rawTexts = Array.from(pres).map((p) => p.textContent?.trim());
    expect(rawTexts).not.toContain("");
  });

  it("renders stream stdout output", () => {
    const { container } = render(
      notebookRenderer.render(minimalNotebook) as React.ReactElement,
    );
    expect(findPreWithText(container, "hello")).toBeDefined();
  });

  it("renders stream stderr output", () => {
    const { container } = render(
      notebookRenderer.render(stderrNotebook) as React.ReactElement,
    );
    const stderrPre = findPreWithText(container, "UserWarning: test");
    expect(stderrPre).toBeDefined();
    expect(stderrPre?.className).toContain("yellow");
  });

  it("renders error output with ename and evalue", () => {
    const { container } = render(
      notebookRenderer.render(multiCellNotebook) as React.ReactElement,
    );
    const errorDiv = container.querySelector(".text-red-400");
    expect(errorDiv).not.toBeNull();
    expect(errorDiv?.textContent).toContain("ValueError");
    expect(errorDiv?.textContent).toContain("oops");
  });

  it("strips ANSI codes from error traceback", () => {
    const { container } = render(
      notebookRenderer.render(multiCellNotebook) as React.ReactElement,
    );
    const errorDiv = container.querySelector(".text-red-400");
    const tracebackPre = errorDiv?.querySelector("pre");
    expect(tracebackPre).not.toBeNull();
    expect(tracebackPre?.textContent).not.toContain("\u001b[");
  });

  it("renders display_data with image/png as img element", () => {
    const { container } = render(
      notebookRenderer.render(displayDataNotebook) as React.ReactElement,
    );
    const img = container.querySelector("img");
    expect(img).not.toBeNull();
    expect(img?.src).toContain("data:image/png;base64,");
    expect(img?.alt).toBe("Cell output");
  });

  it("renders display_data with image/jpeg as img element", () => {
    const { container } = render(
      notebookRenderer.render(jpegOutputNotebook) as React.ReactElement,
    );
    const img = container.querySelector("img");
    expect(img).not.toBeNull();
    expect(img?.src).toContain("data:image/jpeg;base64,");
  });

  it("renders display_data with sanitized image/svg+xml markup", async () => {
    const { container } = render(
      notebookRenderer.render({
        ...svgOutputNotebook,
        cells: [
          {
            ...svgOutputNotebook.cells[0],
            outputs: [
              {
                output_type: "display_data" as const,
                data: {
                  "image/svg+xml":
                    '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script><circle onload="alert(1)" r="10"/></svg>',
                },
              },
            ],
          },
        ],
      }) as React.ReactElement,
    );
    await waitFor(() => expect(container.querySelector("svg")).not.toBeNull());
    expect(container.querySelector("script")).toBeNull();
    expect(container.innerHTML).not.toContain("onload");
  });

  it("renders display_data with sanitized text/html markup", async () => {
    const { container } = render(
      notebookRenderer.render({
        ...htmlOutputNotebook,
        cells: [
          {
            ...htmlOutputNotebook.cells[0],
            outputs: [
              {
                output_type: "display_data" as const,
                data: {
                  "text/html":
                    '<table onclick="alert(1)"><tr><td>cell</td></tr></table><script>alert(1)</script>',
                },
              },
            ],
          },
        ],
      }) as React.ReactElement,
    );
    await waitFor(() =>
      expect(container.querySelector("table")).not.toBeNull(),
    );
    expect(container.querySelector("script")).toBeNull();
    expect(container.innerHTML).not.toContain("onclick");
  });

  it("keeps the output container when sanitization removes all markup", async () => {
    const { container } = render(
      notebookRenderer.render({
        ...htmlOutputNotebook,
        cells: [
          {
            ...htmlOutputNotebook.cells[0],
            outputs: [
              {
                output_type: "display_data" as const,
                data: { "text/html": "<script>alert(1)</script>" },
              },
            ],
          },
        ],
      }) as React.ReactElement,
    );

    await waitFor(() =>
      expect(container.querySelector(".overflow-x-auto")).not.toBeNull(),
    );
    expect(container.querySelector("script")).toBeNull();
  });

  it("does not run DOMPurify during server rendering", () => {
    expect(() => {
      renderToString(
        notebookRenderer.render(htmlOutputNotebook) as React.ReactElement,
      );
      renderToString(
        notebookRenderer.render(svgOutputNotebook) as React.ReactElement,
      );
    }).not.toThrow();
  });

  it("renders execute_result with text/plain fallback", () => {
    const { container } = render(
      notebookRenderer.render(executeResultNotebook) as React.ReactElement,
    );
    expect(findPreWithText(container, "42")).toBeDefined();
  });

  it("defaults to 'python' when no metadata is provided", () => {
    render(notebookRenderer.render(noMetadataNotebook) as React.ReactElement);
    const elements = screen.getAllByText("python");
    expect(elements.length).toBeGreaterThanOrEqual(1);
  });

  it("uses kernelspec language when language_info is absent", () => {
    render(notebookRenderer.render(kernelOnlyNotebook) as React.ReactElement);
    const elements = screen.getAllByText("R");
    expect(elements.length).toBeGreaterThanOrEqual(1);
  });

  it("handles array source (joined)", () => {
    const { container } = render(
      notebookRenderer.render(arraySourceNotebook) as React.ReactElement,
    );
    const codePre = Array.from(container.querySelectorAll("code")).find(
      (el) =>
        el.textContent?.includes("line1") && el.textContent?.includes("line2"),
    );
    expect(codePre).toBeDefined();
  });

  it("handles array text in stream output (joined)", () => {
    const { container } = render(
      notebookRenderer.render(arraySourceNotebook) as React.ReactElement,
    );
    const outputPre = findPreWithText(container, "out1");
    expect(outputPre).toBeDefined();
    expect(outputPre?.textContent).toContain("out2");
  });

  it("renders from JSON string value", () => {
    render(
      notebookRenderer.render(
        JSON.stringify(minimalNotebook),
      ) as React.ReactElement,
    );
    expect(screen.getByText(/python 3\.10\.0/i)).toBeDefined();
    expect(screen.getByText("print('hello')")).toBeDefined();
  });

  it("renders cell count correctly for multiple cells", () => {
    render(notebookRenderer.render(multiCellNotebook) as React.ReactElement);
    expect(screen.getByText("6 cells")).toBeDefined();
  });

  it("returns null for unknown output types", () => {
    expect(() =>
      render(
        notebookRenderer.render(unknownOutputNotebook) as React.ReactElement,
      ),
    ).not.toThrow();
  });

  it("skips empty stream output", () => {
    expect(() =>
      render(
        notebookRenderer.render(emptyStreamNotebook) as React.ReactElement,
      ),
    ).not.toThrow();
  });

  it("returns null for display_data with no matching mime types", () => {
    expect(() =>
      render(
        notebookRenderer.render(
          displayDataNoMatchNotebook,
        ) as React.ReactElement,
      ),
    ).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// CollapsibleOutputs toggle
// ---------------------------------------------------------------------------

describe("notebookRenderer.render collapsible outputs", () => {
  it("shows output by default and hides on toggle", () => {
    const { container } = render(
      notebookRenderer.render(minimalNotebook) as React.ReactElement,
    );

    const toggleBtn = screen.getByRole("button", { name: /hide output/i });
    expect(toggleBtn).toBeDefined();

    // The output container (sibling of toggle button) is visible
    const outputsContainer = container.querySelector(
      ".ml-4.border-l-2 .mt-1.flex.flex-col",
    );
    expect(outputsContainer).not.toBeNull();

    fireEvent.click(toggleBtn);

    expect(screen.getByRole("button", { name: /show output/i })).toBeDefined();
    // The output container is gone after collapse
    const hiddenOutputs = container.querySelector(
      ".ml-4.border-l-2 .mt-1.flex.flex-col",
    );
    expect(hiddenOutputs).toBeNull();
  });

  it("re-shows output after toggling twice", () => {
    const { container } = render(
      notebookRenderer.render(minimalNotebook) as React.ReactElement,
    );

    fireEvent.click(screen.getByRole("button", { name: /hide output/i }));
    expect(
      container.querySelector(".ml-4.border-l-2 .mt-1.flex.flex-col"),
    ).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /show output/i }));
    expect(
      container.querySelector(".ml-4.border-l-2 .mt-1.flex.flex-col"),
    ).not.toBeNull();
  });
});

// ---------------------------------------------------------------------------
// getCopyContent
// ---------------------------------------------------------------------------

describe("notebookRenderer.getCopyContent", () => {
  it("returns JSON string for object value", () => {
    const result = notebookRenderer.getCopyContent(minimalNotebook);
    expect(result).not.toBeNull();
    expect(result!.mimeType).toBe("application/json");
    expect(result!.data).toBe(JSON.stringify(minimalNotebook, null, 2));
    expect(result!.alternativeMimeTypes).toContain("text/plain");
  });

  it("returns original string for string value", () => {
    const jsonStr = JSON.stringify(minimalNotebook);
    const result = notebookRenderer.getCopyContent(jsonStr);
    expect(result).not.toBeNull();
    expect(result!.data).toBe(jsonStr);
  });

  it("provides fallbackText matching the data", () => {
    const result = notebookRenderer.getCopyContent(minimalNotebook);
    expect(result!.fallbackText).toBe(result!.data);
  });
});

// ---------------------------------------------------------------------------
// getDownloadContent
// ---------------------------------------------------------------------------

describe("notebookRenderer.getDownloadContent", () => {
  it("returns blob with correct mimeType", () => {
    const result = notebookRenderer.getDownloadContent(minimalNotebook);
    expect(result).not.toBeNull();
    expect(result!.mimeType).toBe("application/x-ipynb+json");
    expect(result!.data).toBeInstanceOf(Blob);
  });

  it("uses filename from metadata", () => {
    const result = notebookRenderer.getDownloadContent(minimalNotebook, {
      filename: "my_analysis.ipynb",
    });
    expect(result!.filename).toBe("my_analysis.ipynb");
  });

  it("falls back to notebook.ipynb when no metadata filename", () => {
    const result = notebookRenderer.getDownloadContent(minimalNotebook);
    expect(result!.filename).toBe("notebook.ipynb");
  });

  it("falls back to notebook.ipynb when metadata has no filename", () => {
    const result = notebookRenderer.getDownloadContent(minimalNotebook, {});
    expect(result!.filename).toBe("notebook.ipynb");
  });

  it("serializes string value directly into blob", () => {
    const jsonStr = JSON.stringify(minimalNotebook);
    const result = notebookRenderer.getDownloadContent(jsonStr);
    expect(result).not.toBeNull();
    expect(result!.data).toBeInstanceOf(Blob);
  });
});

// ---------------------------------------------------------------------------
// isConcatenable
// ---------------------------------------------------------------------------

describe("notebookRenderer.isConcatenable", () => {
  it("always returns false", () => {
    expect(notebookRenderer.isConcatenable(minimalNotebook)).toBe(false);
    expect(notebookRenderer.isConcatenable("anything")).toBe(false);
    expect(notebookRenderer.isConcatenable(null)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Renderer metadata
// ---------------------------------------------------------------------------

describe("notebookRenderer metadata", () => {
  it("has the correct name", () => {
    expect(notebookRenderer.name).toBe("NotebookRenderer");
  });

  it("has priority 36", () => {
    expect(notebookRenderer.priority).toBe(36);
  });
});
