import { render, screen } from "@/tests/integrations/test-utils";
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
    ChatMessagesContainer: () => <div data-testid="messages" />,
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

const baseProps = {
  messages: [],
  status: "ready",
  error: undefined,
  sessionId: "s1",
  isLoadingSession: false,
  isCreatingSession: false,
  onCreateSession: vi.fn(),
  onSend: vi.fn(),
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
        expertIdentity={{ ...mariaIdentity, isArchived: true }}
      />,
    );

    expect(screen.getByTestId("archived-expert-notice").textContent).toContain(
      "Maria was let go — this thread is read-only",
    );
    expect(screen.queryByTestId("composer")).toBeNull();
  });

  it("keeps the composer for an active expert", () => {
    render(<ChatContainer {...baseProps} expertIdentity={mariaIdentity} />);

    expect(screen.getByTestId("composer")).toBeDefined();
    expect(screen.queryByTestId("archived-expert-notice")).toBeNull();
  });
});
