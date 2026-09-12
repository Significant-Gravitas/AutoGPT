import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import type { ChainRow } from "../helpers";
import { ToolResult } from "../ToolResult";

function row(
  tool: string,
  output: unknown,
  state: ChainRow["state"] = "done",
): ChainRow {
  return { key: tool, category: "team", text: tool, state, tool, output };
}

const REVIEWER = {
  id: "exp-ada",
  name: "Ada",
  role: "Chief of Staff",
  avatar_url: null,
  color: "",
};

describe("consult_teammate verdict card", () => {
  it("shows a pass verdict with its reason", () => {
    render(
      <ToolResult
        row={row("consult_teammate", {
          type: "team_consult",
          message: "Ada ruled: PASS.",
          verdict: "pass",
          reason: "Every commitment is covered by the authority given.",
          quotes: [],
          reviewer: REVIEWER,
        })}
      />,
    );

    expect(screen.getByText("Ada")).toBeDefined();
    expect(screen.getByText("Chief of Staff")).toBeDefined();
    expect(screen.getByText("No objection")).toBeDefined();
    expect(
      screen.getByText("Every commitment is covered by the authority given."),
    ).toBeDefined();
  });

  it("shows a block verdict with the offending lines quoted", () => {
    render(
      <ToolResult
        row={row("consult_teammate", {
          type: "team_consult",
          message: "Ada ruled: BLOCK.",
          verdict: "block",
          reason: "Two commitments are not covered by the authority given.",
          quotes: [
            "The duplicate June charge is refunded.",
            "will be fixed by Friday, 2026-06-13.",
          ],
          reviewer: REVIEWER,
        })}
      />,
    );

    expect(screen.getByText("Blocked")).toBeDefined();
    expect(
      screen.getByText("The duplicate June charge is refunded."),
    ).toBeDefined();
    expect(
      screen.getByText("will be fixed by Friday, 2026-06-13."),
    ).toBeDefined();
  });

  it("shows an insufficient verdict with no quotes rendered", () => {
    render(
      <ToolResult
        row={row("consult_teammate", {
          type: "team_consult",
          message: "Ada could not be reached.",
          verdict: "insufficient",
          reason: "Ada could not be reached for a verdict.",
          quotes: [],
          reviewer: REVIEWER,
        })}
      />,
    );

    expect(screen.getByText("Not checked")).toBeDefined();
    expect(screen.queryByRole("list")).toBeNull();
  });

  it("renders nothing while the tool is still running with no output yet", () => {
    const { container } = render(
      <ToolResult row={row("consult_teammate", undefined, "running")} />,
    );

    expect(container.textContent).toBe("");
  });

  it("renders nothing for an unrecognised verdict", () => {
    const { container } = render(
      <ToolResult
        row={row("consult_teammate", {
          type: "team_consult",
          message: "malformed",
          verdict: "maybe",
          reviewer: REVIEWER,
        })}
      />,
    );

    expect(container.textContent).toBe("");
  });

  it("renders nothing without a reviewer", () => {
    const { container } = render(
      <ToolResult
        row={row("consult_teammate", {
          type: "team_consult",
          message: "malformed",
          verdict: "pass",
        })}
      />,
    );

    expect(container.textContent).toBe("");
  });
});
