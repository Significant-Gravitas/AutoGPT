import { cleanup, render, waitFor } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

const mermaidPlugin = vi.hoisted(() => ({
  loads: 0,
  getMermaid: vi.fn(),
  render: vi.fn(async (_id: string, _source: string) => ({
    svg: '<svg data-testid="diagram"><text>A to B</text></svg>',
  })),
}));

// @streamdown/mermaid imports the whole mermaid library at the top of its
// module, so loading it is loading mermaid.
vi.mock("@streamdown/mermaid", () => {
  mermaidPlugin.loads += 1;
  return {
    mermaid: {
      name: "mermaid",
      type: "diagram",
      language: "mermaid",
      getMermaid(config?: unknown) {
        mermaidPlugin.getMermaid(config);
        return { initialize: vi.fn(), render: mermaidPlugin.render };
      },
    },
  };
});

// Streamdown draws a diagram once its block scrolls into view.
class VisibleIntersectionObserver {
  constructor(
    private readonly callback: (
      entries: { isIntersecting: boolean }[],
      observer: VisibleIntersectionObserver,
    ) => void,
  ) {}
  observe() {
    this.callback([{ isIntersecting: true }], this);
  }
  unobserve() {}
  disconnect() {}
  takeRecords() {
    return [];
  }
}

const DIAGRAM_REPLY = [
  "Here is the flow.",
  "",
  "```mermaid",
  "graph TD",
  "  A --> B",
  "```",
].join("\n");

describe("Streamdown's mermaid plugin in chat replies", () => {
  beforeAll(() => {
    vi.stubGlobal("IntersectionObserver", VisibleIntersectionObserver);
  });

  afterEach(() => {
    cleanup();
  });

  it("does not load mermaid for a reply without a diagram", async () => {
    const { MessageResponse } = await import(
      "@/components/organisms/Message/Message"
    );
    const { container } = render(
      <MessageResponse>{"Just **prose** here."}</MessageResponse>,
    );

    await waitFor(() => expect(container.textContent).toContain("prose"));
    expect(mermaidPlugin.loads).toBe(0);
  });

  it("loads mermaid and draws the diagram once a reply holds one", async () => {
    const { MessageResponse } = await import(
      "@/components/organisms/Message/Message"
    );
    const { findByTestId } = render(
      <MessageResponse>{DIAGRAM_REPLY}</MessageResponse>,
    );

    expect(await findByTestId("diagram", {}, { timeout: 5000 })).toBeTruthy();
    expect(mermaidPlugin.loads).toBe(1);
    expect(mermaidPlugin.render).toHaveBeenCalledWith(
      expect.any(String),
      expect.stringContaining("graph TD\n  A --> B"),
    );
  });

  it("loads mermaid only once and passes Streamdown's config through", async () => {
    const { mermaid } = await import("../streamdown-mermaid-plugin");
    const config = { theme: "dark" as const };

    const [first, second] = await Promise.all([
      mermaid.getMermaid(config).render("first", "graph TD\n  A --> B"),
      mermaid.getMermaid().render("second", "graph LR\n  C --> D"),
    ]);

    expect(mermaidPlugin.loads).toBe(1);
    expect(mermaidPlugin.getMermaid).toHaveBeenCalledWith(config);
    expect(mermaidPlugin.render).toHaveBeenCalledWith(
      "second",
      "graph LR\n  C --> D",
    );
    expect(first.svg).toContain("diagram");
    expect(second.svg).toContain("diagram");
  });
});
