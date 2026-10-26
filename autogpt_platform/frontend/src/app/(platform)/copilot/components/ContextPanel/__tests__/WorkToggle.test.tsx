import { getGetV2GetSessionMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { SessionDetailResponseMessagesItem } from "@/app/api/__generated__/models/sessionDetailResponseMessagesItem";
import { server } from "@/mocks/mock-server";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { WorkToggle } from "../components/WorkToggle/WorkToggle";

function heldHandoff(id: string) {
  return [
    {
      role: "assistant",
      content: "",
      tool_calls: [
        {
          id,
          function: {
            name: "delegate_to_expert",
            arguments: JSON.stringify({ expert_id: "Alex", prompt: "PRD" }),
          },
        },
      ],
    },
    {
      role: "tool",
      tool_call_id: id,
      content: JSON.stringify({
        type: "approval_required",
        review_id: `rev-${id}`,
      }),
    },
  ];
}

function sessionWith(messages: SessionDetailResponseMessagesItem[]) {
  server.use(
    getGetV2GetSessionMockHandler200({
      id: "chat-1",
      created_at: "2026-09-28T00:00:00Z",
      updated_at: "2026-09-28T00:00:00Z",
      user_id: "u-1",
      chat_status: "idle",
      messages,
    }),
  );
}

describe("Work toggle", () => {
  afterEach(() => {
    cleanup();
    useCopilotUIStore.setState((s) => ({
      artifactPanel: { ...s.artifactPanel, isOpen: false, activeTab: "files" },
    }));
  });

  it("counts the hand-offs waiting on the user", async () => {
    sessionWith([...heldHandoff("c1"), ...heldHandoff("c2")]);
    render(<WorkToggle sessionId="chat-1" />);
    const badge = await screen.findByTestId("work-badge");
    expect(badge.textContent).toBe("2");
  });

  it("opens the Work tab and closes it again", async () => {
    sessionWith(heldHandoff("c1"));
    render(<WorkToggle sessionId="chat-1" />);
    fireEvent.click(await screen.findByRole("button", { name: /open work/i }));
    expect(useCopilotUIStore.getState().artifactPanel).toMatchObject({
      isOpen: true,
      activeTab: "work",
    });
    fireEvent.click(screen.getByRole("button", { name: /hide work/i }));
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(false);
  });

  it("stays hidden while the chat has handed nothing off", () => {
    render(<WorkToggle sessionId={null} />);
    expect(screen.queryByRole("button", { name: /work/i })).toBeNull();
    expect(screen.queryByTestId("work-badge")).toBeNull();
  });
});
