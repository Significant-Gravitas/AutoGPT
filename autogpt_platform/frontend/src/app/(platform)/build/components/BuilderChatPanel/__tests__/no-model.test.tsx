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
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { delay, http, HttpResponse } from "msw";
import { parseAsString, useQueryStates } from "nuqs";
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

// The URL has to keep the agent the panel creates, so the testing adapter
// remembers updates and keeps its queue. Even so, once an agent has been
// created the URL can briefly read "no agent" again under load and the panel
// creates a second one; dev's panel does the same. Counts taken after a
// successful creation are therefore lower bounds. Exact counts are kept where
// the point is that nothing was created.
function renderPanel({
  queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  }),
  searchParams,
  beside,
}: {
  queryClient?: QueryClient;
  searchParams?: string;
  beside?: ReactNode;
} = {}) {
  return render(
    <>
      {beside}
      <BuilderChatPanel />
    </>,
    {
      wrapper: ({ children }: { children: ReactNode }) => (
        <QueryClientProvider client={queryClient}>
          <NuqsTestingAdapter
            hasMemory
            resetUrlUpdateQueueOnMount={false}
            searchParams={searchParams}
          >
            <BackendAPIProvider>
              <OnboardingProvider>
                <TooltipProvider>{children}</TooltipProvider>
              </OnboardingProvider>
            </BackendAPIProvider>
          </NuqsTestingAdapter>
        </QueryClientProvider>
      ),
    },
  );
}

// Opens another agent the way the builder does: through the URL.
function OpenAgentButton({ id }: { id: string }) {
  const [, setQuery] = useQueryStates({ flowID: parseAsString });
  return (
    <button type="button" onClick={() => void setQuery({ flowID: id })}>
      Open {id}
    </button>
  );
}

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

    renderPanel();
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

    renderPanel({ searchParams: "?flowID=graph-saved" });
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

    renderPanel();
    fireEvent.click(screen.getByLabelText("Chat with builder"));

    expect(
      await screen.findByText("Could not check the AI connection"),
    ).toBeDefined();
    expect(createdGraphs).toBe(0);

    transportsUp = true;
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(
      await screen.findByPlaceholderText(
        "Ask the builder to edit or run this workflow…",
      ),
    ).toBeDefined();
    expect(createdGraphs).toBeGreaterThanOrEqual(1);
    expect(createdSessions).toBeGreaterThanOrEqual(1);
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

    renderPanel({ queryClient });
    fireEvent.click(screen.getByLabelText("Chat with builder"));

    await expectNoModelNotice();
    expect(createdGraphs).toBe(0);
    expect(createdSessions).toBe(0);
  });

  it("after a failed agent creation, waits for Retry even when the model check refetches", async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    let attempts = 0;
    server.use(
      http.get("/api/proxy/api/chat/transports", async () => {
        await delay(20);
        return HttpResponse.json({
          transports: [{ ...SELF_HOSTED_CHAT, available: true, default: true }],
        });
      }),
      http.post("/api/proxy/api/graphs", () => {
        attempts += 1;
        return attempts === 1
          ? HttpResponse.json({ detail: "boom" }, { status: 500 })
          : HttpResponse.json({
              ...getPostV1CreateNewGraphResponseMock(),
              id: "graph-new",
            });
      }),
    );

    renderPanel({ queryClient });
    fireEvent.click(screen.getByLabelText("Chat with builder"));
    expect(
      await screen.findByText("Could not create a blank workflow"),
    ).toBeDefined();

    // A window focus refetches the transports; that alone must not retry.
    await queryClient.refetchQueries({
      queryKey: getGetV2ListChatTransportsQueryKey(),
    });
    await act(() => new Promise((resolve) => setTimeout(resolve, 50)));
    expect(attempts).toBe(1);
    expect(screen.getByText("Could not create a blank workflow")).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(
      await screen.findByPlaceholderText(
        "Ask the builder to edit or run this workflow…",
      ),
    ).toBeDefined();
    expect(attempts).toBeGreaterThanOrEqual(2);
  });

  it("a session that failed for one agent doesn't hold up the next agent's", async () => {
    server.use(
      getGetV2ListChatTransportsMockHandler({
        transports: [{ ...SELF_HOSTED_CHAT, available: true, default: true }],
      }),
      http.post("/api/proxy/api/chat/sessions", async ({ request }) => {
        const body = (await request.json()) as { builder_graph_id?: string };
        return body.builder_graph_id === "graph-a"
          ? HttpResponse.json({ detail: "boom" }, { status: 500 })
          : HttpResponse.json({
              id: "session-b",
              created_at: "2026-10-12T00:00:00Z",
              user_id: "user-1",
            });
      }),
    );

    renderPanel({
      searchParams: "?flowID=graph-a",
      beside: <OpenAgentButton id="graph-b" />,
    });
    fireEvent.click(screen.getByLabelText("Chat with builder"));
    expect(
      await screen.findByText("Could not start the builder chat"),
    ).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: "Open graph-b" }));
    expect(
      await screen.findByPlaceholderText(
        "Ask the builder to edit or run this workflow…",
      ),
    ).toBeDefined();
    expect(screen.queryByText("Could not start the builder chat")).toBeNull();
  });

  it("still creates the agent and opens the chat when a model is connected", async () => {
    server.use(
      getGetV2ListChatTransportsMockHandler({
        transports: [{ ...SELF_HOSTED_CHAT, available: true, default: true }],
      }),
    );

    renderPanel();
    fireEvent.click(screen.getByLabelText("Chat with builder"));

    expect(
      await screen.findByPlaceholderText(
        "Ask the builder to edit or run this workflow…",
      ),
    ).toBeDefined();
    expect(createdGraphs).toBeGreaterThanOrEqual(1);
    expect(createdSessions).toBeGreaterThanOrEqual(1);
    expect(screen.queryByRole("status", { name: "No AI connection" })).toBe(
      null,
    );
  });
});
