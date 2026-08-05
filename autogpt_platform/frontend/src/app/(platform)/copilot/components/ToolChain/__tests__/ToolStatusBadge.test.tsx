import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { ToolStatusBadge } from "../ToolStatusBadge";

describe("ToolStatusBadge", () => {
  afterEach(cleanup);

  it("morphs declaratively from a spinner into completion", () => {
    const { rerender } = render(
      <ToolStatusBadge state="running" label="Running" morphToCheck>
        <span>tool</span>
      </ToolStatusBadge>,
    );

    expect(screen.getByRole("img", { name: "Running" }).dataset.state).toBe(
      "spinning",
    );

    rerender(
      <ToolStatusBadge state="done" label="Done" morphToCheck>
        <span>tool</span>
      </ToolStatusBadge>,
    );

    expect(screen.getByRole("img", { name: "Done" }).dataset.state).toBe(
      "done",
    );
  });

  it("keeps action-required rows on their tool icon", () => {
    render(
      <ToolStatusBadge state="done" label="Connect" morphToCheck={false}>
        <span>integration</span>
      </ToolStatusBadge>,
    );

    expect(screen.getByRole("img", { name: "Connect" }).dataset.state).toBe(
      "done",
    );
    expect(screen.getByText("integration")).toBeDefined();
  });
});
