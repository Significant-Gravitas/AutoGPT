import { render, screen, fireEvent } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MessagePart } from "../../ChatMessagesContainer/helpers";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { useCopilotUIStore } from "../../../store";
import { DelegationStatusLine } from "../DelegationStatusLine";

function part(
  tool: string,
  id: string,
  output: Record<string, unknown> | undefined,
  input: Record<string, unknown> = {
    expert_id: "exp-alex",
    prompt: "Draft the PRD",
  },
): MessagePart {
  return {
    type: `tool-${tool}`,
    state: output ? "output-available" : "input-available",
    toolCallId: id,
    input,
    output,
  } as unknown as MessagePart;
}

const ALEX = { id: "exp-alex", name: "Alex", role: "Product Manager" };

describe("DelegationStatusLine", () => {
  beforeEach(() => {
    useCopilotUIStore.setState((s) => ({
      artifactPanel: { ...s.artifactPanel, isOpen: false, activeTab: "files" },
    }));
  });
  afterEach(cleanup);

  it("renders nothing for a turn without hand-offs", () => {
    render(
      <DelegationStatusLine
        parts={[part("web_search", "c1", { results: [] }, { query: "x" })]}
        messageId="m1"
      />,
    );
    expect(screen.queryByTestId("delegation-status-line")).toBeNull();
  });

  it("names the expert and the timing once they reported back", () => {
    render(
      <DelegationStatusLine
        parts={[
          part("delegate_to_expert", "c1", {
            status: "completed",
            sub_session_id: "sub-1",
            elapsed_seconds: 400,
            expert: ALEX,
            sub_workspace_files: [{ name: "prd.md", path: "/p" }],
          }),
        ]}
        messageId="m1"
      />,
    );
    const line = screen.getByTestId("delegation-status-line");
    expect(line.getAttribute("data-status")).toBe("completed");
    expect(screen.getByText("Alex reported back")).toBeDefined();
    expect(screen.getByText("· 6m 40s · 1 file")).toBeDefined();
  });

  it("opens the Work tab from the line", () => {
    render(
      <DelegationStatusLine
        parts={[
          part("delegate_to_expert", "c1", {
            status: "completed",
            expert: ALEX,
          }),
        ]}
        messageId="m1"
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /open/i }));
    expect(useCopilotUIStore.getState().artifactPanel.activeTab).toBe("work");
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);
  });

  it("keeps a held hand-off off the line while it waits for approval", () => {
    render(
      <DelegationStatusLine
        parts={[
          part("delegate_to_expert", "c1", {
            type: "approval_required",
            review_id: "rev-1",
          }),
        ]}
        messageId="m1"
      />,
    );
    expect(screen.queryByTestId("delegation-status-line")).toBeNull();
  });

  it("offers a retry that asks Otto to hand off again", () => {
    const onSend = vi.fn();
    render(
      <CopilotChatActionsProvider onSend={onSend}>
        <DelegationStatusLine
          parts={[
            part("delegate_to_expert", "c1", {
              type: "error",
              error: "Budget cap reached",
              expert: ALEX,
            }),
          ]}
          messageId="m1"
        />
      </CopilotChatActionsProvider>,
    );
    expect(screen.getByText("Alex stopped")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: /retry/i }));
    expect(onSend).toHaveBeenCalledWith("Please retry the hand-off to Alex.");
  });
});
