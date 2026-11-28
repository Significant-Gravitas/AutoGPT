import { act, cleanup, render } from "@/tests/integrations/test-utils";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getGetV2GetPendingReviewsForChatSessionMockHandler,
  getGetV2GetPendingReviewsForExecutionMockHandler,
} from "@/app/api/__generated__/endpoints/executions/executions.msw";
import { server } from "@/mocks/mock-server";
import { ChatContainer } from "../components/ChatContainer/ChatContainer";

const chainedCallIDs = new Set<string>();

vi.mock(
  "@/app/(platform)/copilot/components/ToolChain/ToolChain",
  async (importActual) => {
    const actual =
      await importActual<
        typeof import("@/app/(platform)/copilot/components/ToolChain/ToolChain")
      >();
    function RecordingToolChain(props: Parameters<typeof actual.ToolChain>[0]) {
      for (const part of props.parts) {
        if ("toolCallId" in part) chainedCallIDs.add(String(part.toolCallId));
      }
      return <actual.ToolChain {...props} />;
    }
    return { ToolChain: RecordingToolChain };
  },
);

vi.mock("@/app/(platform)/copilot/components/ChatInput/ChatInput", () => ({
  ChatInput: () => <textarea aria-label="Chat input" />,
}));

vi.mock(
  "@/app/(platform)/copilot/components/UsageLimits/useIsUsageLimitReached",
  () => ({ useIsUsageLimitReached: () => false }),
);

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => ({
  ...(await importActual<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: () => false,
}));

type Message = UIMessage<unknown, UIDataTypes, UITools>;

/** Every tool that had a component of its own before the tool chain drew
 *  all of them. Each must still reach the chain through the real transcript. */
const TOOLS = [
  "ask_question",
  "find_block",
  "find_capability",
  "describe_capability",
  "run_capability",
  "resume_capability",
  "find_agent",
  "find_library_agent",
  "search_docs",
  "get_doc_page",
  "connect_integration",
  "run_block",
  "continue_run_block",
  "run_mcp_tool",
  "run_agent",
  "schedule_agent",
  "setup_agent_webhook_trigger",
  "decompose_goal",
  "create_agent",
  "edit_agent",
  "view_agent_output",
  "search_feature_requests",
  "create_feature_request",
  "create_folder",
  "list_folders",
  "update_folder",
  "move_folder",
  "delete_folder",
  "move_agents_to_folder",
  "TodoWrite",
];

const STATES = ["input-available", "output-available"] as const;

function transcript(tool: string, state: (typeof STATES)[number]): Message[] {
  return [
    { id: "user-1", role: "user", parts: [{ type: "text", text: "Go ahead" }] },
    {
      id: "assistant-1",
      role: "assistant",
      parts: [
        {
          type: `tool-${tool}`,
          toolCallId: `call-${tool}`,
          state,
          input: {},
          ...(state === "output-available" && { output: {} }),
        } as unknown as Message["parts"][number],
      ],
    },
  ];
}

class MockResizeObserver {
  observe() {}
  disconnect() {}
  unobserve() {}
}

beforeEach(() => {
  chainedCallIDs.clear();
  vi.stubGlobal("ResizeObserver", MockResizeObserver);
  server.use(
    getGetV2GetPendingReviewsForChatSessionMockHandler([]),
    getGetV2GetPendingReviewsForExecutionMockHandler([]),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("tool parts in a reply", () => {
  it.each(TOOLS.flatMap((tool) => STATES.map((state) => [tool, state])))(
    "%s (%s) renders inside the tool chain",
    async (tool, state) => {
      render(
        <ChatContainer
          messages={transcript(tool, state as (typeof STATES)[number])}
          status="ready"
          error={undefined}
          sessionId="chain-session"
          isLoadingSession={false}
          isCreatingSession={false}
          onCreateSession={vi.fn()}
          onSend={vi.fn()}
          onStop={vi.fn()}
        />,
      );
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });

      expect(chainedCallIDs.has(`call-${tool}`)).toBe(true);
    },
  );
});
