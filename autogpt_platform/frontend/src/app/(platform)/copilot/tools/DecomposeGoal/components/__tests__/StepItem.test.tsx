import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import { StepItem } from "../StepItem";

describe("StepItem", () => {
  it("renders step number and description", () => {
    render(
      <StepItem index={0} description="Add input block" status="pending" />,
    );
    expect(screen.getByText("1. Add input block")).toBeDefined();
  });

  it("renders block name when provided", () => {
    render(
      <StepItem
        index={1}
        description="Add AI summarizer"
        blockName="AI Text Generator"
        status="pending"
      />,
    );
    expect(screen.getByText("AI Text Generator")).toBeDefined();
  });

  it("does not render block name when null", () => {
    render(
      <StepItem
        index={0}
        description="Connect blocks"
        blockName={null}
        status="pending"
      />,
    );
    expect(screen.queryByText("AI Text Generator")).toBeNull();
  });

  it("renders pending icon by default", () => {
    render(<StepItem index={0} description="Step" status="pending" />);
    expect(screen.getByLabelText("pending")).toBeDefined();
  });

  it("renders completed icon for completed status", () => {
    render(<StepItem index={0} description="Step" status="completed" />);
    expect(screen.getByLabelText("completed")).toBeDefined();
  });

  it("renders in-progress icon for in_progress status", () => {
    render(<StepItem index={0} description="Step" status="in_progress" />);
    expect(screen.getByLabelText("in progress")).toBeDefined();
  });

  it("renders failed icon for failed status", () => {
    render(<StepItem index={0} description="Step" status="failed" />);
    expect(screen.getByLabelText("failed")).toBeDefined();
  });

  it("uses zero-based index to render 1-based step number", () => {
    render(<StepItem index={4} description="Fifth step" status="pending" />);
    expect(screen.getByText("5. Fifth step")).toBeDefined();
  });
});
