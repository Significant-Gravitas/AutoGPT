import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { describe, expect, it, vi } from "vitest";
import { ChatContainer } from "../ChatContainer";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

vi.mock("@/app/(platform)/copilot/components/ChatInput/ChatInput", () => ({
  ChatInput: () => <div data-testid="composer" />,
}));

vi.mock(
  "@/app/(platform)/copilot/components/ChatMessagesContainer/ChatMessagesContainer",
  () => ({
    ChatMessagesContainer: ({ onRetry }: { onRetry?: () => void }) => (
      <div data-testid="messages">
        <button
          type="button"
          data-testid="error-retry"
          onClick={() => onRetry?.()}
        />
      </div>
    ),
  }),
);

vi.mock(
  "@/app/(platform)/copilot/components/ChatContainer/useAutoOpenArtifacts",
  () => ({
    useAutoOpenArtifacts: () => undefined,
  }),
);

vi.mock(
  "@/app/(platform)/copilot/components/UsageLimits/useIsUsageLimitReached",
  () => ({
    useIsUsageLimitReached: () => false,
  }),
);

const userMessage: UIMessage<unknown, UIDataTypes, UITools> = {
  id: "m1",
  role: "user",
  parts: [{ type: "text", text: "Plan my week" }],
};

const baseProps = {
  messages: [userMessage],
  status: "error",
  error: new Error("boom"),
  sessionId: "s1",
  isLoadingSession: false,
  isCreatingSession: false,
  onCreateSession: vi.fn(),
  onStop: vi.fn(),
};

const mariaIdentity = {
  id: "expert-maria",
  name: "Maria",
  avatarUrl: null,
  role: "Marketing Strategist",
  isArchived: false,
};

describe("ChatContainer — archived expert", () => {
  it("replaces the composer with a read-only notice for a fired expert", () => {
    render(
      <ChatContainer
        {...baseProps}
        onSend={vi.fn()}
        expertIdentity={{ ...mariaIdentity, isArchived: true }}
      />,
    );

    expect(screen.getByTestId("archived-expert-notice").textContent).toContain(
      "Maria was let go — this thread is read-only",
    );
    expect(screen.queryByTestId("composer")).toBeNull();
  });

  it("keeps the composer for an active expert", () => {
    render(
      <ChatContainer
        {...baseProps}
        onSend={vi.fn()}
        expertIdentity={mariaIdentity}
      />,
    );

    expect(screen.getByTestId("composer")).toBeDefined();
    expect(screen.queryByTestId("archived-expert-notice")).toBeNull();
  });

  it("never resends through error retry on an archived thread", () => {
    const onSend = vi.fn();
    render(
      <ChatContainer
        {...baseProps}
        onSend={onSend}
        expertIdentity={{ ...mariaIdentity, isArchived: true }}
      />,
    );

    fireEvent.click(screen.getByTestId("error-retry"));

    expect(onSend).not.toHaveBeenCalled();
  });

  it("retries the last user message on an active thread", () => {
    const onSend = vi.fn();
    render(
      <ChatContainer
        {...baseProps}
        onSend={onSend}
        expertIdentity={mariaIdentity}
      />,
    );

    fireEvent.click(screen.getByTestId("error-retry"));

    expect(onSend).toHaveBeenCalledWith("Plan my week");
  });
});
