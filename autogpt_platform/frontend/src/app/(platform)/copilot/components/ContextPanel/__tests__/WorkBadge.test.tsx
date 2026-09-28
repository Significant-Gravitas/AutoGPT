import { getGetV2GetSessionMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { PanelTabs } from "../components/PanelTabs";

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

describe("Work tab badge", () => {
  afterEach(cleanup);

  it("counts the hand-offs waiting on the user", async () => {
    server.use(
      getGetV2GetSessionMockHandler200({
        id: "chat-1",
        created_at: "2026-09-28T00:00:00Z",
        updated_at: "2026-09-28T00:00:00Z",
        user_id: "u-1",
        chat_status: "idle",
        messages: [...heldHandoff("c1"), ...heldHandoff("c2")],
      }),
    );
    render(<PanelTabs sessionId="chat-1" />);
    const badge = await screen.findByTestId("work-badge");
    expect(badge.textContent).toBe("2");
  });

  it("shows no badge when nothing waits", () => {
    render(<PanelTabs sessionId={null} />);
    expect(screen.queryByTestId("work-badge")).toBeNull();
  });
});
