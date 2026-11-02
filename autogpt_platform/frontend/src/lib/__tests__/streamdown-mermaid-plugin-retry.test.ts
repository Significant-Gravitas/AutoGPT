import { describe, expect, it, vi } from "vitest";

const mermaidPlugin = vi.hoisted(() => ({
  attempts: 0,
  getMermaid: vi.fn(),
}));

// The first fetch of the chunk fails, as it would offline or mid-deploy.
vi.mock("@streamdown/mermaid", () => {
  mermaidPlugin.attempts += 1;
  if (mermaidPlugin.attempts === 1) throw new Error("chunk failed to load");
  return {
    mermaid: {
      name: "mermaid",
      type: "diagram",
      language: "mermaid",
      getMermaid(config?: unknown) {
        mermaidPlugin.getMermaid(config);
        return {
          initialize: vi.fn(),
          render: async () => ({ svg: "<svg></svg>" }),
        };
      },
    },
  };
});

describe("lazy mermaid plugin after a failed load", () => {
  it("fails that diagram and loads mermaid again for the next one", async () => {
    const { mermaid } = await import("../streamdown-mermaid-plugin");

    await expect(
      mermaid.getMermaid().render("first", "graph TD\n  A --> B"),
    ).rejects.toThrow();

    const instance = mermaid.getMermaid();
    instance.initialize({ theme: "forest" });
    await expect(
      instance.render("second", "graph TD\n  A --> B"),
    ).resolves.toEqual({ svg: "<svg></svg>" });
    expect(mermaidPlugin.attempts).toBe(2);
    expect(mermaidPlugin.getMermaid).toHaveBeenCalledWith({ theme: "forest" });
  });
});
