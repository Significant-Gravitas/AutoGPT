import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import { WorkCard } from "../WorkCard";
import { type WorkRunMetadata } from "../helpers";

const baseMeta: WorkRunMetadata = {
  executionId: "exec-1",
  graphId: "graph-1",
  libraryAgentId: "lib-1",
  graphName: "Weekly Report",
  status: "completed",
  outputType: "unknown",
};

describe("WorkCard", () => {
  it("renders the title, completed chip and preview", () => {
    render(<WorkCard metadata={baseMeta} preview="Sent the weekly report." />);

    expect(screen.getByText("Weekly Report")).toBeDefined();
    expect(screen.getByText("Completed")).toBeDefined();
    expect(screen.getByText("Sent the weekly report.")).toBeDefined();
  });

  it("shows a Failed chip for failed runs", () => {
    render(
      <WorkCard metadata={{ ...baseMeta, status: "failed" }} preview="" />,
    );
    expect(screen.getByText("Failed")).toBeDefined();
  });

  it("opens the output sheet when Open is clicked", () => {
    render(<WorkCard metadata={baseMeta} preview="Sent the weekly report." />);

    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    expect(
      screen.getByRole("link", { name: "Open run details" }),
    ).toBeDefined();
  });
});
