import { getGetV2GetSessionMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MessagePart } from "../../ChatMessagesContainer/helpers";
import { HeldOutcomesContext } from "../../ChatMessagesContainer/HeldOutcomesContext";
import { SupersededDelegationsContext } from "../../../supersededDelegationsContext";
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

  it("shows the cost when the run reports one", () => {
    render(
      <DelegationStatusLine
        parts={[
          part("delegate_to_expert", "c1", {
            status: "completed",
            elapsed_seconds: 400,
            cost_usd: 0.31,
            expert: ALEX,
          }),
        ]}
        messageId="m1"
      />,
    );
    expect(screen.getByText("· 6m 40s · $0.31")).toBeDefined();
  });

  it("counts a queued teammate and names who starts when free", () => {
    render(
      <DelegationStatusLine
        parts={[
          part("delegate_to_expert", "c1", {
            status: "queued",
            expert: ALEX,
          }),
        ]}
        messageId="m1"
      />,
    );
    expect(screen.getByText("1 expert queued")).toBeDefined();
    expect(
      screen.getByText("· Alex starts as soon as they are free"),
    ).toBeDefined();
  });

  it("leaves the approval state once the approved run's result lands", () => {
    render(
      <HeldOutcomesContext.Provider
        value={
          new Map([
            [
              "c1",
              {
                outcome: "approved",
                output: { status: "queued", sub_session_id: "", expert: ALEX },
              },
            ],
          ])
        }
      >
        <DelegationStatusLine
          parts={[
            part("delegate_to_expert", "c1", {
              type: "approval_required",
              review_id: "rev-1",
            }),
          ]}
          messageId="m1"
        />
      </HeldOutcomesContext.Provider>,
    );
    const line = screen.getByTestId("delegation-status-line");
    expect(line.getAttribute("data-status")).toBe("queued");
  });

  it("shows a completed hand-off as working again once its thread runs again", async () => {
    server.use(
      getGetV2GetSessionMockHandler200({
        id: "sub-1",
        created_at: "2026-09-28T10:00:00Z",
        updated_at: "2026-09-28T10:00:00Z",
        user_id: "u-1",
        chat_status: "running",
        messages: [],
      }),
    );
    const done = part("delegate_to_expert", "c1", {
      status: "completed",
      sub_session_id: "sub-1",
      expert: ALEX,
    });
    const { unmount } = render(
      <DelegationStatusLine parts={[done]} messageId="m1" />,
    );
    await waitFor(() =>
      expect(
        screen
          .getByTestId("delegation-status-line")
          .getAttribute("data-status"),
      ).toBe("running"),
    );
    unmount();

    // A later re-delegation owns the thread now: this run stays done.
    render(
      <SupersededDelegationsContext.Provider value={new Set(["c1"])}>
        <DelegationStatusLine parts={[done]} messageId="m1" />
      </SupersededDelegationsContext.Provider>,
    );
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(
      screen.getByTestId("delegation-status-line").getAttribute("data-status"),
    ).toBe("completed");
  });

  it("says the user stopped the teammate after a stopped turn reloads", () => {
    render(
      <DelegationStatusLine
        parts={[
          {
            ...part("delegate_to_expert", "c1", undefined),
            state: "output-available",
            output: "",
          } as unknown as MessagePart,
        ]}
        messageId="m1"
      />,
    );
    const line = screen.getByTestId("delegation-status-line");
    expect(line.getAttribute("data-status")).toBe("cancelled");
    expect(screen.getByText("You stopped exp-alex")).toBeDefined();
  });

  it("needs the user when the teammate stopped on a question, whatever the transcript says", async () => {
    server.use(
      getGetV2GetSessionMockHandler200({
        id: "sub-1",
        created_at: "2026-09-28T10:00:00Z",
        updated_at: "2026-09-28T10:00:00Z",
        user_id: "u-1",
        chat_status: "idle",
        active_stream: null,
        metadata: {
          pending_question: {
            text: "Q4 or December?",
            asked_at: new Date("2026-09-28T10:03:00Z"),
          },
        },
        messages: [
          { role: "user", content: "Draft the PRD" },
          {
            role: "assistant",
            content: "",
            tool_calls: [
              {
                id: "ask-1",
                function: {
                  name: "ask_question",
                  arguments: JSON.stringify({
                    questions: [
                      { question: "Q4 or December?", options: ["Q4", "Dec"] },
                    ],
                  }),
                },
              },
            ],
          },
        ],
      }),
    );
    render(
      <DelegationStatusLine
        parts={[
          part("delegate_to_expert", "c1", {
            status: "completed",
            sub_session_id: "sub-1",
            response: "Q4 or December?",
            expert: ALEX,
          }),
        ]}
        messageId="m1"
      />,
    );
    expect(await screen.findByText("Alex needs you")).toBeDefined();
    expect(
      screen.getByTestId("delegation-status-line").getAttribute("data-status"),
    ).toBe("needs-input");
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
