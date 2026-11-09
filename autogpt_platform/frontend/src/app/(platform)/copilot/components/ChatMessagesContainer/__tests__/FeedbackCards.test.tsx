import {
  getGetV2GetFeedbackPromptMockHandler200,
  getPatchV2UpdateFeedbackPromptMockHandler200,
} from "@/app/api/__generated__/endpoints/feedback-prompts/feedback-prompts.msw";
import { getPostV2SubmitMessageFeedbackMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import type { ComponentProps, ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatMessagesContainer } from "../ChatMessagesContainer";
import { isChainableToolPart } from "../helpers";
import type { ExpertIdentity } from "../../../useExpertMap";

type Message = UIMessage<unknown, UIDataTypes, UITools>;

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: () => false,
}));
vi.mock("use-stick-to-bottom", () => ({
  useStickToBottomContext: () => ({
    scrollRef: {
      current: { scrollHeight: 100, scrollTop: 0, clientHeight: 500 },
    },
  }),
}));
vi.mock("@/components/organisms/Conversation/Conversation", () => ({
  Conversation: ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  ),
  ConversationContent: ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  ),
  ConversationScrollButton: () => null,
}));
vi.mock("../components/AssistantMessageActions", () => ({
  AssistantMessageActions: () => null,
}));
vi.mock("../../JobStatsBar/TurnStatsBar", () => ({ TurnStatsBar: () => null }));
vi.mock("../../CopilotPendingReviews/CopilotPendingReviews", () => ({
  CopilotPendingReviews: () => null,
}));
vi.mock("../../ChatMinimap/ChatMinimap", () => ({ ChatMinimap: () => null }));
vi.mock("../../DrawnLineLoader/DrawnLineLoader", () => ({
  DrawnLineLoader: () => null,
}));

class MockIntersectionObserver {
  observe() {}
  disconnect() {}
  unobserve() {}
  takeRecords() {
    return [];
  }
  root = null;
  rootMargin = "";
  thresholds = [];
}

const MARIA: ExpertIdentity = {
  id: "maria",
  name: "Maria",
  avatarUrl: null,
  role: null,
  isArchived: false,
  readOnlyReason: null,
};

function toolPart(
  type: string,
  output: Record<string, unknown>,
): Message["parts"][number] {
  return {
    type,
    toolCallId: `call-${type}`,
    state: "output-available",
    input: {},
    output: { type: "feedback_prompt", message: "", ...output },
  } as Message["parts"][number];
}

function workCheck(promptID: string | null = "prompt-1", isFirst = true) {
  return toolPart("tool-request_work_check", {
    kind: "work_check",
    prompt_id: promptID,
    is_first: isFirst,
  });
}

function mistake(promptID: string | null = "prompt-2") {
  return toolPart("tool-flag_mistake", {
    kind: "otto_mistake",
    prompt_id: promptID,
    is_first: false,
  });
}

function reply(parts: Message["parts"]): Message {
  return { id: "sess-1-seq-4", role: "assistant", parts };
}

const USER: Message = {
  id: "sess-1-seq-3",
  role: "user",
  parts: [{ type: "text", text: "Do the thing" }],
};

const FOLLOW_UP: Message = {
  id: "sess-1-seq-5",
  role: "user",
  parts: [{ type: "text", text: "One more thing" }],
};

const baseProps = {
  messages: [USER] as Message[],
  status: "ready",
  error: undefined,
  isLoading: false,
  sessionID: "sess-1",
  queuedMessages: [] as string[],
  hasMoreMessages: false,
  isLoadingMore: false,
  onLoadMore: vi.fn(),
  onRetry: vi.fn(),
} satisfies ComponentProps<typeof ChatMessagesContainer>;

function setup(
  state: Partial<{
    answered: boolean;
    dismissed: boolean;
    score: number | null;
  }> = {},
) {
  const ratings: Record<string, unknown>[] = [];
  const patches: unknown[] = [];
  server.use(
    getGetV2GetFeedbackPromptMockHandler200({
      id: "prompt-1",
      kind: "work_check",
      answered: false,
      dismissed: false,
      reason: null,
      score: null,
      ...state,
    }),
    getPatchV2UpdateFeedbackPromptMockHandler200(async ({ request }) => {
      patches.push(await request.clone().json());
      return {
        id: "prompt-1",
        kind: "work_check",
        answered: true,
        dismissed: false,
        reason: null,
        score: null,
      };
    }),
    getPostV2SubmitMessageFeedbackMockHandler200(async ({ request }) => {
      ratings.push((await request.clone().json()) as Record<string, unknown>);
      return { id: "feedback-1", score_name: "work_check", score_value: 0 };
    }),
  );
  return { ratings, patches };
}

describe("feedback cards in a chat", () => {
  beforeEach(() => {
    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("asks about the task under the last settled reply, as Otto by default", async () => {
    setup();
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={[USER, reply([workCheck()])]}
      />,
    );
    expect(await screen.findByText("How did Otto do?")).toBeDefined();
    expect(
      screen.getByText("Quick check-in on your first finished task"),
    ).toBeDefined();
  });

  it("uses the expert's name in an expert chat and the weekly wording", async () => {
    setup();
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={[USER, reply([workCheck("prompt-1", false)])]}
        expertIdentity={MARIA}
      />,
    );
    expect(await screen.findByText("How did Maria do?")).toBeDefined();
    expect(screen.getByText("Quick check-in on this task")).toBeDefined();
  });

  it("stays hidden while the reply streams", async () => {
    setup();
    render(
      <ChatMessagesContainer
        {...baseProps}
        status="streaming"
        messages={[USER, reply([workCheck()])]}
      />,
    );
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.queryByText("How did Otto do?")).toBeNull();
  });

  it("goes away once the user follows up", async () => {
    setup();
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={[USER, reply([workCheck()]), FOLLOW_UP]}
      />,
    );
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.queryByText("How did Otto do?")).toBeNull();
  });

  it("renders nothing when the backend said not to ask", async () => {
    setup();
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={[USER, reply([workCheck(null), mistake(null)])]}
      />,
    );
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.queryByText("How did Otto do?")).toBeNull();
    expect(screen.queryByText("Help the team fix this")).toBeNull();
  });

  it("shows the thanks line for an answered check after a reload", async () => {
    setup({ answered: true, score: 2 });
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={[USER, reply([workCheck()])]}
      />,
    );
    expect(
      await screen.findByText(/Glad to hear it. We'll pass that on to Otto/),
    ).toBeDefined();
  });

  it("shows nothing for a closed check after a reload", async () => {
    setup({ dismissed: true });
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={[USER, reply([workCheck()])]}
      />,
    );
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.queryByText("How did Otto do?")).toBeNull();
  });

  it("posts Great as the top work check score", async () => {
    const { ratings, patches } = setup();
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={[USER, reply([workCheck()])]}
      />,
    );
    await screen.findByText("How did Otto do?");
    fireEvent.click(screen.getByRole("button", { name: "Great" }));
    await waitFor(() =>
      expect(ratings).toEqual([
        {
          message_id: "sess-1-seq-4",
          score_name: "work_check",
          score_value: 2,
        },
      ]),
    );
    await waitFor(() => expect(patches).toEqual([{ score: 2 }]));
  });

  it("sends a picked reason with the rating", async () => {
    const { ratings } = setup();
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={[USER, reply([workCheck()])]}
      />,
    );
    await screen.findByText("How did Otto do?");
    fireEvent.click(screen.getByRole("button", { name: "Not good" }));
    fireEvent.click(
      await screen.findByRole("button", { name: "Got facts wrong" }),
    );
    await waitFor(() =>
      expect(ratings).toContainEqual(
        expect.objectContaining({
          score_name: "work_check",
          score_value: 0,
          reason: "wrong_facts",
        }),
      ),
    );
  });

  it("shows the mistake card as Otto even in an expert chat", async () => {
    const { ratings } = setup();
    render(
      <ChatMessagesContainer
        {...baseProps}
        messages={[USER, reply([mistake()])]}
        expertIdentity={MARIA}
      />,
    );
    expect(await screen.findByText("Help the team fix this")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Made things up" }));
    await waitFor(() =>
      expect(ratings).toContainEqual(
        expect.objectContaining({
          score_name: "otto_mistake",
          score_value: 0,
          reason: "made_up",
        }),
      ),
    );
  });

  it("keeps both cards out of the tool chain", () => {
    expect(isChainableToolPart(workCheck())).toBe(false);
    expect(isChainableToolPart(mistake())).toBe(false);
  });
});
