import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { parseWorkspace } from "@/lib/openui/parse";
import { showResult } from "./renderConnected";

function proseView(markdown: string) {
  return `root = Workspace("Trip notes", "Written notes", [notes])\nnotes = Prose(${JSON.stringify(markdown)})`;
}

describe("Prose sections in chat", () => {
  it("renders Markdown with the chat renderer", async () => {
    await showResult(
      undefined,
      false,
      proseView(
        "## Packing\n\nBring **both** adapters.\n\n- Passport\n- Charger",
      ),
    );
    expect(
      await screen.findByRole("heading", { name: "Packing" }),
    ).toBeDefined();
    expect(screen.getByText("both").dataset.streamdown).toBe("strong");
    expect(screen.getByText("Passport").closest("li")).not.toBeNull();
    expect(screen.getByText("Charger").closest("li")).not.toBeNull();
  });

  it("does not render raw HTML from the model", async () => {
    const view = await showResult(
      undefined,
      false,
      proseView(
        'Before <b>bold html</b> after\n\n<img src="x" onerror="alert(1)">\n\n<script>alert(1)</script>',
      ),
    );
    await screen.findByText(/Before/);
    expect(view.container.querySelector("b")).toBeNull();
    expect(view.container.querySelector("img")).toBeNull();
    expect(view.container.querySelector("script")).toBeNull();
  });

  it("rejects Prose longer than 8000 characters", () => {
    expect(parseWorkspace(proseView("a".repeat(8000))).typeName).toBe(
      "Workspace",
    );
    expect(() => parseWorkspace(proseView("a".repeat(8001)))).toThrow();
  });
});
