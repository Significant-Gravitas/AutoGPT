import { getGetV2GetCopilotUsageMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getGetSubscriptionStatusMockHandler200 } from "@/app/api/__generated__/endpoints/credits/credits.msw";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { ChatContainer, type ChatContainerProps } from "../ChatContainer";

const voice = vi.hoisted(() => ({ isActive: false }));

vi.mock("../../../voice/useVoiceMode", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../../voice/useVoiceMode")>()),
  useVoiceMode: () => ({
    state: voice.isActive ? "listening" : "off",
    isActive: voice.isActive,
    isStarting: false,
    statusLabel: "",
    toggle: vi.fn(),
    failure: null,
    retryFailedUtterance: vi.fn(),
    downloadFailedUtterance: vi.fn(),
  }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

const props: ChatContainerProps = {
  messages: [],
  status: "ready",
  error: undefined,
  sessionId: "empty-session",
  isLoadingSession: false,
  isCreatingSession: false,
  onCreateSession: vi.fn(),
  onStop: vi.fn(),
  onSend: vi.fn(),
  expertIdentity: {
    id: "expert-kepler",
    name: "Kepler",
    role: "researcher",
    avatarUrl: null,
    isArchived: false,
    readOnlyReason: null,
  },
};

// The chat locks on the AutoPilot budget, and onboarding stays hidden while it
// does: pin a budget with room left instead of faker's random usage and tier.
beforeEach(() => {
  voice.isActive = false;
  server.use(
    getGetV2GetCopilotUsageMockHandler200({
      tier: "PRO",
      daily: null,
      weekly: null,
      reset_cost: 0,
      state: "ok",
      limiting_window: null,
      resets_at: null,
    }),
    getGetSubscriptionStatusMockHandler200({
      tier: "PRO",
      monthly_cost: 0,
      tier_costs: {},
      proration_credit_cents: 0,
    }),
  );
});

describe("new full-page expert chats", () => {
  test("an empty chat opens the chat whose setup questions still wait", async () => {
    const request = mockOnboarding();
    const urlUpdates = renderAt(<ChatContainer {...props} />, "empty-session");
    await waitFor(() => expect(urlUpdates).toEqual(["kickoff-session"]));
    expect(request).toHaveBeenCalledOnce();
  });

  test("stays in the empty chat once setup is answered or skipped", async () => {
    const request = mockOnboarding(true);
    const urlUpdates = renderAt(<ChatContainer {...props} />, "empty-session");
    await waitFor(() => expect(request).toHaveBeenCalled());
    expect(urlUpdates).toEqual([]);
  });

  test("stays in a new chat while its first message's files upload", async () => {
    const request = mockOnboarding();
    const urlUpdates = renderAt(
      <ChatContainer
        {...props}
        isUploadingFiles
        pendingSend={{ text: "Here's the brief", attachments: [] }}
      />,
      "empty-session",
    );
    expect(await screen.findByRole("textbox")).toBeDefined();
    expect(request).not.toHaveBeenCalled();
    expect(urlUpdates).toEqual([]);
  });

  test("stays in a new chat while voice mode runs in it", async () => {
    voice.isActive = true;
    const request = mockOnboarding();
    const urlUpdates = renderAt(<ChatContainer {...props} />, "empty-session");
    expect(await screen.findByRole("textbox")).toBeDefined();
    expect(request).not.toHaveBeenCalled();
    expect(urlUpdates).toEqual([]);
  });

  test("keeps the home hero when an expert is picked before a chat exists", async () => {
    const request = mockOnboarding();
    render(<ChatContainer {...props} sessionId={null} />);
    expect(await screen.findByRole("textbox")).toBeDefined();
    expect(request).not.toHaveBeenCalled();
  });

  test("does not request onboarding after the user sends a prompt", async () => {
    const request = mockOnboarding();
    render(
      <ChatContainer
        {...props}
        sessionId="existing-session"
        messages={[
          {
            id: "user-1",
            role: "user",
            parts: [{ type: "text", text: "Research this company" }],
          },
        ]}
      />,
    );
    expect(await screen.findByText("Research this company")).toBeDefined();
    expect(request).not.toHaveBeenCalled();
  });
});

function mockOnboarding(completed = false) {
  const request = vi.fn(() =>
    HttpResponse.json(completed ? null : { session_id: "kickoff-session" }),
  );
  server.use(http.get("/api/proxy/api/experts/:expertId/onboarding", request));
  return request;
}

function renderAt(ui: ReactNode, sessionId: string) {
  const urlUpdates: (string | null)[] = [];
  const Wrapper = withNuqsTestingAdapter({
    searchParams: `?sessionId=${sessionId}`,
    hasMemory: true,
    onUrlUpdate: (event) => {
      urlUpdates.push(event.searchParams.get("sessionId"));
    },
  });
  render(<Wrapper>{ui}</Wrapper>);
  return urlUpdates;
}
