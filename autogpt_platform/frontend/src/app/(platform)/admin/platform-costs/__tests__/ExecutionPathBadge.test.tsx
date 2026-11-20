import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import { ExecutionPathBadge } from "../components/ExecutionPathBadge";

describe("ExecutionPathBadge", () => {
  it("shows a dash when the run has no execution path", () => {
    render(<ExecutionPathBadge executionPath={null} />);
    expect(screen.getByText("—").className).toContain(
      "text-foreground-tertiary",
    );
  });

  it("shows a sync run as a neutral badge", () => {
    render(<ExecutionPathBadge executionPath="sync_baseline" />);
    const badge = screen.getByTitle("sync_baseline");
    expect(badge.textContent).toBe("sync");
    expect(badge.className).toContain("bg-surface-hover");
  });

  it("shows a batch run as a success badge", () => {
    render(<ExecutionPathBadge executionPath="openai_batch" />);
    const badge = screen.getByTitle("openai_batch");
    expect(badge.textContent).toBe("batch");
    expect(badge.className).toContain("text-success-foreground");
  });
});
