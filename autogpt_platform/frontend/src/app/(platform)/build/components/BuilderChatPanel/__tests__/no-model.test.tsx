import { getGetV2ListChatTransportsQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import {
  getGetV2ListChatTransportsMockHandler,
  getPostV2CreateSessionMockHandler,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getPostV1CreateNewGraphMockHandler,
  getPostV1CreateNewGraphResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import type { ChatTransportResponse } from "@/app/api/__generated__/models/chatTransportResponse";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { server } from "@/mocks/mock-server";
import OnboardingProvider from "@/providers/onboarding/onboarding-provider";
import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { delay, http, HttpResponse } from "msw";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it } from "vitest";
import { BuilderChatPanel } from "../BuilderChatPanel";

// A self-hosted install with no model: chat can never answer, so opening the
// builder chat must say how to fix that and must not leave anything behind.

const SELF_HOSTED_CHAT: ChatTransportResponse = {
  auth_provider: "platform",
  credential_id: null,
  label: "Self-hosted chat",
  available: false,
  default: false,
};

let createdGraphs: number;
let createdSessions: number;

beforeEach(() => {
  createdGraphs = 0;
  createdSessions = 0;
  server.use(
    getPostV1CreateNewGraphMockHandler(() => {
      createdGraphs += 1;
      return { ...getPostV1CreateNewGraphResponseMock(), id: "graph-new" };
    }),
    getPostV2CreateSessionMockHandler(() => {
      createdSessions += 1;
      return {
        id: "session-1",
        created_at: "2026-10-12T00:00:00Z",
        user_id: "user-1",
      };
    }),
  );
});

async function expectNoModelNotice() {
  const notice = await screen.findByRole("status", {
    name: "No AI connection",
  });
  expect(notice.textContent).toContain("No AI model is connected");
  expect(
    within(notice)
      .getByRole("link", { name: /setup guide/i })
      .getAttribute("href"),
  ).toContain("single-container.md#models-and-memory");
  expect(screen.queryByRole("button", { name: /Retry/i })).toBeNull();
  expect(screen.queryByText(/Something went wrong/i)).toBeNull();
}

describe("Builder chat with no AI model connected", () => {
  it("says how to connect one and creates no agent and no session", async () => {
    server.use(
      getGetV2ListChatTransportsMockHandler({ transports: [SELF_HOSTED_CHAT] }),
    );

    render(<BuilderChatPanel />);
    fireEvent.click(screen.getByLabelText("Chat with builder"));

    await expectNoModelNotice();
    expect(createdGraphs).toBe(0);
    expect(createdSessions).toBe(0);
  });

  it("falls back to the server's answer when the model check itself fails", async () => {
    server.use(
      http.get("/api/proxy/api/chat/transports", () =>
        HttpResponse.json({ detail: "unavailable" }, { status: 500 }),
      ),
      http.post("/api/proxy/api/chat/sessions", () => {
        createdSessions += 1;
        return HttpResponse.json(
          { detail: "chat_transport_not_configured" },
          { status: 503 },
        );
      }),
    );

    render(
      <NuqsTestingAdapter searchParams="?flowID=graph-saved">
        <BuilderChatPanel />
      </NuqsTestingAdapter>,
    );
    fireEvent.click(screen.getByLabelText("Chat with builder"));

    await expectNoModelNotice();
    expect(createdSessions).toBe(1);
    expect(createdGraphs).toBe(0);
    expect(screen.queryByText(/Could not start the builder chat/i)).toBeNull();
  });

  it("with no agent open, a failed model check creates nothing and offers a retry", async () => {
    let transportsUp = false;
    server.use(
      http.get("/api/proxy/api/chat/transports", () =>
        transportsUp
          ? HttpResponse.json({
              transports: [
                { ...SELF_HOSTED_CHAT, available: true, default: true },
              ],
            })
          : HttpResponse.json({ detail: "unavailable" }, { status: 500 }),
      ),
    );

    render(<BuilderChatPanel />);
    fireEvent.click(screen.getByLabelText("Chat with builder"));

    expect(
      await screen.findByText("Could not check the AI connection"),
    ).toBeDefined();
    expect(createdGraphs).toBe(0);

    transportsUp = true;
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(
      await screen.findByPlaceholderText(
        "Ask the builder to edit or run this agent…",
      ),
    ).toBeDefined();
    expect(createdGraphs).toBe(1);
    expect(createdSessions).toBe(1);
  });

  it("re-checks a model it saw earlier before creating an agent", async () => {
    // The model was there when this tab last asked, and has gone since.
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    queryClient.setQueryData(getGetV2ListChatTransportsQueryKey(), {
      status: 200,
      data: {
        transports: [{ ...SELF_HOSTED_CHAT, available: true, default: true }],
      },
      headers: new Headers(),
    });
    server.use(
      http.get("/api/proxy/api/chat/transports", async () => {
        await delay(50);
        return HttpResponse.json({ transports: [SELF_HOSTED_CHAT] });
      }),
    );

    render(<BuilderChatPanel />, {
      wrapper: ({ children }: { children: ReactNode }) => (
        <QueryClientProvider client={queryClient}>
          <NuqsTestingAdapter>
            <BackendAPIProvider>
              <OnboardingProvider>
                <TooltipProvider>{children}</TooltipProvider>
              </OnboardingProvider>
            </BackendAPIProvider>
          </NuqsTestingAdapter>
        </QueryClientProvider>
      ),
    });
    fireEvent.click(screen.getByLabelText("Chat with builder"));

    await expectNoModelNotice();
    expect(createdGraphs).toBe(0);
    expect(createdSessions).toBe(0);
  });

  it("still creates the agent and opens the chat when a model is connected", async () => {
    server.use(
      getGetV2ListChatTransportsMockHandler({
        transports: [{ ...SELF_HOSTED_CHAT, available: true, default: true }],
      }),
    );

    render(<BuilderChatPanel />);
    fireEvent.click(screen.getByLabelText("Chat with builder"));

    expect(
      await screen.findByPlaceholderText(
        "Ask the builder to edit or run this agent…",
      ),
    ).toBeDefined();
    expect(createdGraphs).toBe(1);
    expect(createdSessions).toBe(1);
    expect(screen.queryByRole("status", { name: "No AI connection" })).toBe(
      null,
    );
  });
});
