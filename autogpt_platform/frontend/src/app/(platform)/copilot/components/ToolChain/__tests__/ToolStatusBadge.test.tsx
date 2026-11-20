import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { ToolStatusBadge } from "../ToolStatusBadge";

describe("ToolStatusBadge", () => {
  afterEach(cleanup);

  it("keeps the tool icon when running completes", () => {
    const { rerender } = render(
      <ToolStatusBadge state="running" label="Running">
        <span>tool</span>
      </ToolStatusBadge>,
    );

    expect(screen.getByRole("img", { name: "Running" }).dataset.state).toBe(
      "spinning",
    );
    expect(screen.getByText("tool")).toBeDefined();

    rerender(
      <ToolStatusBadge state="done" label="Done">
        <span>tool</span>
      </ToolStatusBadge>,
    );

    expect(screen.getByRole("img", { name: "Done" }).dataset.state).toBe(
      "done",
    );
    expect(screen.getByText("tool")).toBeDefined();
  });

  it("keeps completed rows on their tool icon", () => {
    render(
      <ToolStatusBadge state="done" label="Connect">
        <span>integration</span>
      </ToolStatusBadge>,
    );

    expect(screen.getByRole("img", { name: "Connect" }).dataset.state).toBe(
      "done",
    );
    expect(screen.getByText("integration")).toBeDefined();
  });
});
