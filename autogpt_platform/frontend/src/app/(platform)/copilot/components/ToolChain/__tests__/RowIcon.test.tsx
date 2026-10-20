import { cleanup, fireEvent, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { ChainRow } from "../helpers";
import { ProviderIcon, RowIcon } from "../RowIcon";

function row(overrides: Partial<ChainRow>): ChainRow {
  return {
    key: "row",
    category: "other",
    text: "Row",
    state: "done",
    ...overrides,
  };
}

describe("RowIcon", () => {
  afterEach(cleanup);

  it("renders the red alert icon for errored rows regardless of category", () => {
    const { container } = render(
      <RowIcon row={row({ category: "web", state: "error" })} />,
    );

    expect(container.querySelector("svg.text-red-500")).not.toBeNull();
    expect(container.querySelector("svg.text-zinc-600")).toBeNull();
  });

  it.each([
    "narration",
    "reasoning",
    "bash",
    "web",
    "browser",
    "file-read",
    "file-write",
    "file-delete",
    "file-list",
    "search",
    "edit",
    "todo",
    "compaction",
    "agent",
    "agent-build",
    "plan",
    "block",
    "memory",
    "folder",
    "schedule",
    "trigger",
    "preset",
    "chat",
    "mcp",
    "docs",
    "skill",
    "integration",
    "feature",
    "question",
    "info",
    "other",
  ] as const)("renders a themed icon for the %s category", (category) => {
    const { container } = render(<RowIcon row={row({ category })} />);

    expect(container.querySelector("svg.text-zinc-600")).not.toBeNull();
    expect(container.querySelector("svg.text-red-500")).toBeNull();
  });
});

describe("ProviderIcon", () => {
  afterEach(cleanup);

  it("renders the provider image from the given source", () => {
    const { container } = render(
      <ProviderIcon
        src="/integrations/github.png"
        row={row({ category: "integration" })}
      />,
    );

    const image = container.querySelector("img");
    expect(image?.getAttribute("src")).toContain("/integrations/github.png");
  });

  it("falls back to the category icon when the image fails to load", () => {
    const { container } = render(
      <ProviderIcon
        src="/integrations/missing.png"
        row={row({ category: "integration" })}
      />,
    );

    const image = container.querySelector("img");
    expect(image).not.toBeNull();
    if (image) fireEvent.error(image);

    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("svg.text-zinc-600")).not.toBeNull();
  });
});
