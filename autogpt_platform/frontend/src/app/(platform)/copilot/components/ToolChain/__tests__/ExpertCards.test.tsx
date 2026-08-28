import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import type { ChainRow } from "../helpers";
import { ToolResult } from "../ToolResult";

function row(tool: string, output: unknown): ChainRow {
  return {
    key: tool,
    category: "team",
    text: tool,
    state: "done",
    tool,
    output,
  };
}

describe("expert change cards", () => {
  it("shows the proposed charter and that nothing happened yet", () => {
    render(
      <ToolResult
        row={row("raise_expert", {
          type: "expert_change_proposed",
          message: "Nothing created yet.",
          applied: false,
          confirmation_id: "c-1",
          preview: {
            kind: "raise",
            name: "Otto",
            role: "Inbox triage",
            about: "You group the morning inbox.",
            boundaries: "You never send a reply yourself.",
            weekly_budget: 2000,
          },
        })}
      />,
    );

    expect(screen.getByText("Otto")).toBeDefined();
    expect(screen.getByText("Inbox triage")).toBeDefined();
    expect(screen.getByText("Needs your OK")).toBeDefined();
    expect(
      screen.getByText("Stops at: You never send a reply yourself."),
    ).toBeDefined();
    expect(screen.getByText("Weekly budget: 2000 credits")).toBeDefined();
    expect(screen.queryByRole("link", { name: /Edit/ })).toBeNull();
    expect(screen.queryByRole("link", { name: /Chat/ })).toBeNull();
  });

  it("offers Edit and Chat once the expert exists", () => {
    render(
      <ToolResult
        row={row("confirm_expert_change", {
          type: "expert_change_applied",
          message: "Otto is on the team.",
          applied: true,
          kind: "raise",
          expert: {
            id: "exp-otto",
            name: "Otto",
            role: "Inbox triage",
            avatar_url: null,
            color: "violet",
          },
        })}
      />,
    );

    expect(screen.getByText("Otto")).toBeDefined();
    expect(screen.queryByText("Needs your OK")).toBeNull();
    // The applied state must be readable, not colour-and-glyph only.
    expect(screen.getByText("Hired")).toBeDefined();
    expect(
      screen.getByRole("link", { name: /Edit/ }).getAttribute("href"),
    ).toBe("/team/exp-otto");
    expect(
      screen.getByRole("link", { name: /Chat/ }).getAttribute("href"),
    ).toBe("/copilot?expertId=exp-otto");
  });

  it("names the workflows that failed to install on a partial hire", () => {
    render(
      <ToolResult
        row={row("confirm_expert_change", {
          type: "expert_change_applied",
          message: "Otto is on the team.",
          applied: true,
          kind: "hire",
          expert: { id: "exp-otto", name: "Otto", role: "Inbox triage" },
          failed_workflows: ["Inbox sweep", "Weekly digest"],
        })}
      />,
    );

    expect(screen.getByText("Hired")).toBeDefined();
    expect(
      screen.getByText(/Couldn't set up: Inbox sweep, Weekly digest/),
    ).toBeDefined();
  });

  it("stays quiet when every workflow installed", () => {
    render(
      <ToolResult
        row={row("confirm_expert_change", {
          type: "expert_change_applied",
          message: "Otto is on the team.",
          applied: true,
          kind: "hire",
          expert: { id: "exp-otto", name: "Otto", role: "Inbox triage" },
          failed_workflows: [],
        })}
      />,
    );

    expect(screen.queryByText(/Couldn't set up/)).toBeNull();
  });

  it("renders every applied roster member as Hired", () => {
    render(
      <ToolResult
        row={row("confirm_expert_change", {
          type: "expert_change_batch_applied",
          message: "Team ready: Otto, Scout.",
          applied: true,
          results: [
            {
              confirmation_id: "c-1",
              outcome: "applied",
              kind: "hire",
              expert: { id: "exp-otto", name: "Otto", role: "Inbox" },
            },
            {
              confirmation_id: "c-2",
              outcome: "applied",
              kind: "raise",
              expert: { id: "exp-scout", name: "Scout", role: "Research" },
            },
          ],
          experts: [],
        })}
      />,
    );

    expect(screen.getByText("Otto")).toBeDefined();
    expect(screen.getByText("Scout")).toBeDefined();
    expect(screen.getAllByText("Hired")).toHaveLength(2);
  });

  it("never exposes internal failure instructions or ids", () => {
    render(
      <ToolResult
        row={row("confirm_expert_change", {
          type: "expert_change_batch_applied",
          message: "Internal result",
          applied: false,
          results: [
            {
              confirmation_id: "private-id",
              proposed_name: "Scout",
              outcome: "failed",
              reason: "expired",
              error: "Call hire_expert again with confirmation_id private-id.",
            },
          ],
          experts: [],
        })}
      />,
    );

    expect(screen.getByText("Scout: Not added")).toBeDefined();
    expect(
      screen.getByText("This preview expired before approval."),
    ).toBeDefined();
    expect(screen.queryByText(/private-id/)).toBeNull();
    expect(screen.queryByText(/hire_expert/)).toBeNull();
  });

  it("renders a handoff as the receiving teammate's sub-session", () => {
    render(
      <ToolResult
        row={{
          ...row("handoff_to_expert", {
            type: "mcp_tool_output",
            message: "Bea picked it up.",
            status: "running",
            sub_session_id: "sub-1",
            sub_autopilot_session_link: "/copilot?sessionId=sub-1",
            expert: { id: "exp-b", name: "Bea", role: "Ops lead" },
          }),
          category: "agent",
        }}
      />,
    );

    expect(screen.getByText("Bea")).toBeDefined();
    expect(screen.getByText("Ops lead")).toBeDefined();
    expect(screen.getByLabelText("Open sub-session").getAttribute("href")).toBe(
      "/copilot?sessionId=sub-1",
    );
  });
});
