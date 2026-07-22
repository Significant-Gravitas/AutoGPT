import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import { SharedMemorySection } from "./SharedMemorySection";

describe("SharedMemorySection", () => {
  it("renders the shared-memory card for org admins", () => {
    render(<SharedMemorySection isAdmin />);

    expect(screen.getByTestId("org-shared-memory-section")).toBeDefined();
    expect(
      screen.getByRole("heading", { name: "Shared memory" }),
    ).toBeDefined();
  });

  it("renders nothing for non-admins", () => {
    render(<SharedMemorySection isAdmin={false} />);

    expect(screen.queryByTestId("org-shared-memory-section")).toBeNull();
  });

  it("shows the hold-for-review toggle disabled and flags the review queue as blocked", () => {
    render(<SharedMemorySection isAdmin />);

    const toggle = screen.getByRole("switch", {
      name: "Hold new memories for review",
    });
    // No org settings endpoint exists to persist this yet, so the control is
    // rendered disabled rather than wired to an invented endpoint.
    expect(toggle.hasAttribute("disabled")).toBe(true);
    expect(screen.getByTestId("org-memory-review-blocked")).toBeDefined();
  });
});
