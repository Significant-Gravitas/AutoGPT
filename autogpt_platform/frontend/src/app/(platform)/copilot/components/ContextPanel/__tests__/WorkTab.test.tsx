import { getGetV2GetSessionMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { WorkTab } from "../components/WorkTab/WorkTab";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

const ALEX = {
  id: "exp-alex",
  name: "Alex",
  role: "Product Manager",
  avatar_url: null,
  color: "",
};

function delegateCall(id: string, prompt: string) {
  return {
    role: "assistant",
    content: "",
    tool_calls: [
      {
        id,
        function: {
          name: "delegate_to_expert",
          arguments: JSON.stringify({ expert_id: "exp-alex", prompt }),
        },
      },
    ],
  };
}

function toolOutput(id: string, output: Record<string, unknown>) {
  return { role: "tool", tool_call_id: id, content: JSON.stringify(output) };
}

function chatSession(messages: Record<string, unknown>[]) {
  return getGetV2GetSessionMockHandler200({
    id: "chat-1",
    created_at: "2026-09-28T00:00:00Z",
    updated_at: "2026-09-28T00:00:00Z",
    user_id: "u-1",
    chat_status: "idle",
    messages,
  });
}

describe("WorkTab", () => {
  afterEach(cleanup);

  it("shows the empty state for a chat without hand-offs", async () => {
    server.use(chatSession([{ role: "user", content: "hi" }]));
    render(<WorkTab sessionId="chat-1" />);
    expect(await screen.findByText("Nothing delegated yet")).toBeDefined();
  });

  it("lists each hand-off with who, status and what came back", async () => {
    server.use(
      chatSession([
        { role: "user", content: "Draft the PRD" },
        delegateCall("call-1", "Draft the PRD for the onboarding revamp"),
        toolOutput("call-1", {
          type: "mcp_tool_output",
          status: "completed",
          sub_session_id: "sub-1",
          sub_autopilot_session_link: "/copilot?sessionId=sub-1",
          elapsed_seconds: 400,
          response: "Draft attached, three open questions.",
          expert: ALEX,
          sub_workspace_files: [
            { name: "prd.md", path: "/sessions/sub-1/prd.md" },
          ],
        }),
      ]),
    );
    render(<WorkTab sessionId="chat-1" />);

    const row = await screen.findByTestId("delegation-row");
    expect(row.getAttribute("data-status")).toBe("completed");
    expect(screen.getByText("Alex")).toBeDefined();
    expect(screen.getByText("Product Manager")).toBeDefined();
    expect(screen.getByText("Reported back · 1 file")).toBeDefined();
    expect(screen.getByText("6m 40s")).toBeDefined();
    expect(screen.getByText("Otto → 1 hand-off")).toBeDefined();
  });

  it("opens a hand-off's detail and returns to the list", async () => {
    server.use(
      chatSession([
        delegateCall("call-1", "Draft the PRD for the onboarding revamp"),
        toolOutput("call-1", {
          status: "completed",
          sub_session_id: "sub-1",
          sub_autopilot_session_link: "/copilot?sessionId=sub-1",
          response: "Draft attached.",
          expert: ALEX,
        }),
      ]),
    );
    render(<WorkTab sessionId="chat-1" />);

    fireEvent.click(await screen.findByTestId("delegation-row"));
    const detail = await screen.findByTestId("delegation-detail");
    expect(detail.textContent).toContain(
      "Draft the PRD for the onboarding revamp",
    );
    expect(screen.getByText("What came back")).toBeDefined();
    expect(screen.getByText("Draft attached.")).toBeDefined();
    expect(
      screen.getByRole("link", { name: /open thread/i }).getAttribute("href"),
    ).toBe("/copilot?sessionId=sub-1");

    fireEvent.click(screen.getByText("All work"));
    expect(await screen.findByTestId("delegation-row")).toBeDefined();
  });

  it("shows a stopped hand-off with its error", async () => {
    server.use(
      chatSession([
        delegateCall("call-1", "Draft the PRD"),
        toolOutput("call-1", {
          type: "error",
          error: "Weekly budget cap reached",
        }),
      ]),
    );
    render(<WorkTab sessionId="chat-1" />);

    const row = await screen.findByTestId("delegation-row");
    expect(row.getAttribute("data-status")).toBe("failed");
    expect(screen.getByText("Weekly budget cap reached")).toBeDefined();
    expect(screen.getByText("Failed")).toBeDefined();
  });
});
