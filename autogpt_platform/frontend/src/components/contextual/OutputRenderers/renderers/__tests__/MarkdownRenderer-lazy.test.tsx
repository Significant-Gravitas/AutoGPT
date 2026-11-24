import { render, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { markdownRenderer, preloadMarkdownContent } from "../MarkdownRenderer";

// The shared setup hands tests the renderer loaded; this file sees the real wait.
vi.unmock("@/components/contextual/OutputRenderers/renderers/MarkdownRenderer");

describe("markdown output before its renderer has loaded", () => {
  it("shows the text, then formats it once the renderer is here", async () => {
    const { container } = render(
      <>{markdownRenderer.render("## Tides\n\nHigh water at **06:12**.")}</>,
    );

    expect(container.textContent).toBe("## Tides\n\nHigh water at **06:12**.");
    expect(container.querySelector("h2")).toBeNull();

    await preloadMarkdownContent();
    await waitFor(() =>
      expect(container.querySelector("h2")?.textContent).toBe("Tides"),
    );
    expect(container.querySelector("strong")?.textContent).toBe("06:12");
  });
});
