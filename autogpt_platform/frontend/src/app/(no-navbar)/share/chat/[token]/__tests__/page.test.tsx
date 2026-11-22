import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { waitFor } from "@testing-library/react";
import { onlineManager, QueryClient } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";

import {
  getGetV2GetSharedChatMessagesMockHandler200,
  getGetV2GetSharedChatMessagesMockHandler404,
  getGetV2GetSharedChatMessagesMockHandler422,
  getGetV2GetSharedChatMockHandler200,
  getGetV2GetSharedChatMockHandler404,
  getGetV2GetSharedChatMockHandler422,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { SharedChatMessagesPage } from "@/app/api/__generated__/models/sharedChatMessagesPage";
import type { SharedChatSession } from "@/app/api/__generated__/models/sharedChatSession";
import { resetClientRequestFailureLog } from "@/app/api/mutators/request-failure-log";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { server } from "@/mocks/mock-server";
import {
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import SharedChatPage from "../page";

const mockUseParams = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useParams: mockUseParams,
  usePathname: () => "/share/chat/test-token",
  useRouter: () => ({
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
    push: vi.fn(),
    refresh: vi.fn(),
    replace: vi.fn(),
  }),
  useSearchParams: () => new URLSearchParams(),
}));

const TOKEN = "550e8400-e29b-41d4-a716-446655440000";

function resetArtifactPanelStore() {
  useCopilotUIStore.setState({
    artifactPanel: {
      isOpen: false,
      activeArtifact: null,
      history: [],
      activeTab: "files",
      lastArtifact: null,
      mode: "artifact",
      computer: null,
      isComputerOpen: false,
    },
  });
  useCopilotUIStore.getState().resetAutoOpenState();
}

beforeEach(() => {
  vi.clearAllMocks();
  mockUseParams.mockReturnValue({ token: TOKEN });
  resetArtifactPanelStore();
});

afterEach(() => {
  resetClientRequestFailureLog();
});

function deployChat(): SharedChatSession {
  return {
    id: "session-123",
    title: "How to deploy",
    created_at: new Date("2026-05-12T00:00:00Z"),
    updated_at: new Date("2026-05-12T00:00:00Z"),
    shared_at: new Date("2026-05-12T00:00:00Z"),
    linked_executions: [],
  };
}

function deployChatMessages(): SharedChatMessagesPage {
  return {
    messages: [
      {
        id: "m1",
        role: "user",
        content: "How do I deploy?",
        name: null,
        tool_call_id: null,
        tool_calls: null,
        function_call: null,
        sequence: 0,
        created_at: new Date("2026-05-12T00:00:01Z"),
      },
    ],
    has_more: false,
    oldest_sequence: 0,
  };
}

// The app's client keeps React Query's three retries; this one only drops the
// backoff between them, so a test can count them without waiting ~7 s.
function createRetryingQueryClient() {
  return new QueryClient({ defaultOptions: { queries: { retryDelay: 0 } } });
}

// No generated handler answers a 5xx or a network error, so these two answer
// every read of the chat (or of its messages) with `respond` and count them.
function answerChatWith(respond: () => Response) {
  const reads: string[] = [];
  server.use(
    http.get("*/api/public/shared/chats/:shareToken", ({ params }) => {
      reads.push(String(params.shareToken));
      return respond();
    }),
  );
  return reads;
}

function answerMessagesWith(respond: () => Response) {
  const reads: string[] = [];
  server.use(
    http.get("*/api/public/shared/chats/:shareToken/messages", ({ params }) => {
      reads.push(String(params.shareToken));
      return respond();
    }),
  );
  return reads;
}

function serverError() {
  return HttpResponse.json(
    { detail: "Internal Server Error" },
    { status: 500 },
  );
}

describe("SharedChatPage", () => {
  test("renders the read-only viewer with messages on happy path", async () => {
    server.use(
      getGetV2GetSharedChatMockHandler200(
        (): SharedChatSession => ({
          id: "session-123",
          title: "How to deploy",
          created_at: new Date("2026-05-12T00:00:00Z"),
          updated_at: new Date("2026-05-12T00:00:00Z"),
          shared_at: new Date("2026-05-12T00:00:00Z"),
          linked_executions: [],
        }),
      ),
      getGetV2GetSharedChatMessagesMockHandler200(
        (): SharedChatMessagesPage => ({
          messages: [
            {
              id: "m1",
              role: "user",
              content: "How do I deploy?",
              name: null,
              tool_call_id: null,
              tool_calls: null,
              function_call: null,
              sequence: 0,
              created_at: new Date("2026-05-12T00:00:01Z"),
            },
            {
              id: "m2",
              role: "assistant",
              content: "Use docker compose.",
              name: null,
              tool_call_id: null,
              tool_calls: null,
              function_call: null,
              sequence: 1,
              created_at: new Date("2026-05-12T00:00:02Z"),
            },
          ],
          has_more: false,
          oldest_sequence: 0,
        }),
      ),
    );

    render(<SharedChatPage />);

    // Title sits in the ShareHeader's title slot.  Logo + actions
    // slots also live in the header but aren't asserted here.
    const heading = await screen.findByRole("heading", { level: 1 });
    expect(heading.textContent).toBe("How to deploy");
    expect(screen.getByText(/^Shared /)).toBeDefined();
    expect(await screen.findByText("How do I deploy?")).toBeDefined();
    expect(await screen.findByText("Use docker compose.")).toBeDefined();
  });

  test("centers the loading state across the shared chat chrome", () => {
    server.use(
      getGetV2GetSharedChatMockHandler200(
        (): SharedChatSession => ({
          id: "session-loading",
          title: "Loading",
          created_at: new Date("2026-05-12T00:00:00Z"),
          updated_at: new Date("2026-05-12T00:00:00Z"),
          shared_at: new Date("2026-05-12T00:00:00Z"),
          linked_executions: [],
        }),
      ),
    );

    render(<SharedChatPage />);

    const loadingState = screen.getByTestId("shared-chat-loading-state");
    expect(loadingState.className).toContain("w-full");
    expect(loadingState.className).toContain("flex-1");
  });

  test("surfaces the has_more notice when the chat is truncated", async () => {
    server.use(
      getGetV2GetSharedChatMockHandler200(
        (): SharedChatSession => ({
          id: "session-123",
          title: "Long chat",
          created_at: new Date("2026-05-12T00:00:00Z"),
          updated_at: new Date("2026-05-12T00:00:00Z"),
          shared_at: new Date("2026-05-12T00:00:00Z"),
          linked_executions: [],
        }),
      ),
      getGetV2GetSharedChatMessagesMockHandler200(
        (): SharedChatMessagesPage => ({
          messages: [
            {
              id: "m1",
              role: "assistant",
              content: "trimmed",
              name: null,
              tool_call_id: null,
              tool_calls: null,
              function_call: null,
              sequence: 999,
              created_at: new Date("2026-05-12T00:00:00Z"),
            },
          ],
          has_more: true,
          oldest_sequence: 999,
        }),
      ),
    );

    render(<SharedChatPage />);

    expect(
      await screen.findByText(/showing the most recent .* messages/i),
    ).toBeDefined();
    expect(
      await screen.findByText(/older history is not visible/i),
    ).toBeDefined();
  });

  describe("when the chat can't be read", () => {
    test.each([
      {
        link: "a revoked link (404)",
        token: TOKEN,
        answer: (onRead: (token: string) => void) =>
          getGetV2GetSharedChatMockHandler404(({ params }) => {
            onRead(String(params.shareToken));
          }),
      },
      {
        link: "a truncated link (422)",
        token: TOKEN.slice(0, -4),
        answer: (onRead: (token: string) => void) =>
          getGetV2GetSharedChatMockHandler422(({ params }) => {
            onRead(String(params.shareToken));
            return {
              detail: [
                {
                  type: "string_pattern_mismatch",
                  loc: ["path", "share_token"],
                  msg: "String should match pattern",
                },
              ],
            };
          }),
      },
      {
        link: "an answer with no chat in it (200)",
        token: TOKEN,
        answer: (onRead: (token: string) => void) =>
          http.get("*/api/public/shared/chats/:shareToken", ({ params }) => {
            onRead(String(params.shareToken));
            return HttpResponse.json(null);
          }),
      },
    ])(
      "says 'Share link not found' for $link after one read, with no 'Try again'",
      async ({ token, answer }) => {
        mockUseParams.mockReturnValue({ token });
        const reads: string[] = [];
        server.use(answer((read) => reads.push(read)));

        render(<SharedChatPage />, {
          queryClient: createRetryingQueryClient(),
        });

        expect(
          await screen.findByRole("heading", { name: "Share link not found" }),
        ).toBeDefined();
        expect(screen.getByText(/invalid or has been disabled/i)).toBeDefined();
        expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
        expect(reads).toEqual([token]);
      },
    );

    test.each([
      { failure: "a 500 from the backend", respond: serverError },
      {
        failure: "a 502 from the proxy (backend down)",
        respond: () =>
          HttpResponse.json({ error: "Proxy request failed" }, { status: 502 }),
      },
      { failure: "a network error", respond: () => HttpResponse.error() },
    ])(
      "says 'Unable to load', not 'Share link not found', after retrying $failure",
      async ({ respond }) => {
        const reads = answerChatWith(respond);

        render(<SharedChatPage />, {
          queryClient: createRetryingQueryClient(),
        });

        expect(
          await screen.findByRole("heading", { name: "Unable to load" }),
        ).toBeDefined();
        expect(screen.getByRole("button", { name: "Try again" })).toBeDefined();
        expect(screen.queryByText(/share link not found/i)).toBeNull();
        expect(screen.queryByText(/disabled by the owner/i)).toBeNull();
        // The first read and three retries.
        expect(reads).toEqual([TOKEN, TOKEN, TOKEN, TOKEN]);
      },
    );

    test("shows the chat when 'Try again' gets through", async () => {
      answerChatWith(serverError);
      render(<SharedChatPage />, { queryClient: createRetryingQueryClient() });
      expect(
        await screen.findByRole("heading", { name: "Unable to load" }),
      ).toBeDefined();

      server.use(
        getGetV2GetSharedChatMockHandler200(deployChat),
        getGetV2GetSharedChatMessagesMockHandler200(deployChatMessages),
      );
      fireEvent.click(screen.getByRole("button", { name: "Try again" }));

      expect(await screen.findByText("How do I deploy?")).toBeDefined();
      expect(screen.getByRole("heading", { level: 1 }).textContent).toBe(
        "How to deploy",
      );
      expect(
        screen.queryByRole("heading", { name: "Unable to load" }),
      ).toBeNull();
    });

    test("keeps the chat's header and offers 'Try again' when only its messages fail", async () => {
      server.use(getGetV2GetSharedChatMockHandler200(deployChat));
      const messageReads = answerMessagesWith(serverError);
      render(<SharedChatPage />, { queryClient: createRetryingQueryClient() });

      expect(
        await screen.findByText(
          "Couldn't load messages. The share link is valid.",
        ),
      ).toBeDefined();
      expect(screen.getByRole("heading", { level: 1 }).textContent).toBe(
        "How to deploy",
      );
      expect(screen.queryByText(/share link not found/i)).toBeNull();
      expect(messageReads).toHaveLength(4);

      server.use(
        getGetV2GetSharedChatMessagesMockHandler200(deployChatMessages),
      );
      fireEvent.click(screen.getByRole("button", { name: "Try again" }));

      expect(await screen.findByText("How do I deploy?")).toBeDefined();
      expect(screen.queryByText(/couldn't load messages/i)).toBeNull();
    });

    test.each([
      { status: 404, handler: getGetV2GetSharedChatMessagesMockHandler404 },
      { status: 422, handler: getGetV2GetSharedChatMessagesMockHandler422 },
    ])(
      "says 'Share link not found' when the messages read answers $status after the chat loaded",
      async ({ handler }) => {
        const messageReads: string[] = [];
        server.use(
          getGetV2GetSharedChatMockHandler200(deployChat),
          handler(({ params }) => {
            messageReads.push(String(params.shareToken));
          }),
        );
        render(<SharedChatPage />, {
          queryClient: createRetryingQueryClient(),
        });

        expect(
          await screen.findByRole("heading", { name: "Share link not found" }),
        ).toBeDefined();
        expect(screen.queryByText(/the share link is valid/i)).toBeNull();
        expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
        expect(messageReads).toEqual([TOKEN]);
      },
    );

    test("keeps loading while the visitor is offline, then shows the chat", async () => {
      server.use(
        getGetV2GetSharedChatMockHandler200(deployChat),
        getGetV2GetSharedChatMessagesMockHandler200(deployChatMessages),
      );
      onlineManager.setOnline(false);
      try {
        render(<SharedChatPage />);

        expect(
          await screen.findByTestId("shared-chat-loading-state"),
        ).toBeDefined();
        expect(screen.queryByText(/share link not found/i)).toBeNull();

        onlineManager.setOnline(true);

        expect(await screen.findByText("How do I deploy?")).toBeDefined();
      } finally {
        onlineManager.setOnline(true);
      }
    });
  });

  test("wraps the error state in the branded share header", async () => {
    // Cursor regression: previously the error state returned without
    // ShareHeader, so a revoked link showed a raw card with no logo
    // / no Copy-link / Sign-up CTAs — inconsistent with the execution
    // share viewer.  All states (loading, error, success) must share
    // the branded chrome.
    server.use(
      http.get(
        "*/api/public/shared/chats/:token",
        () => new HttpResponse(null, { status: 404 }),
      ),
    );

    render(<SharedChatPage />);

    await screen.findByText(/share link not found/i);
    // Logo image always present in ShareHeader.
    expect(screen.getAllByAltText("AutoGPT").length).toBeGreaterThan(0);
    // Copy-link CTA wired through ShareActions in the chrome.
    expect(screen.getByRole("button", { name: /copy link/i })).toBeDefined();
  });

  test("uses split layout (flex-row) so ArtifactPanel can dock alongside messages", async () => {
    // Owner-side copilot renders messages + ArtifactPanel side-by-side
    // via ``flex-row``.  The viewer must match — otherwise opening an
    // artifact covers the chat full-screen instead of splitting.
    server.use(
      getGetV2GetSharedChatMockHandler200(
        (): SharedChatSession => ({
          id: "session-split",
          title: "Layout check",
          created_at: new Date("2026-05-12T00:00:00Z"),
          updated_at: new Date("2026-05-12T00:00:00Z"),
          shared_at: new Date("2026-05-12T00:00:00Z"),
          linked_executions: [],
        }),
      ),
      getGetV2GetSharedChatMessagesMockHandler200(
        (): SharedChatMessagesPage => ({
          messages: [],
          has_more: false,
          oldest_sequence: null,
        }),
      ),
    );

    const { container } = render(<SharedChatPage />);
    // Wait for the page to settle past loading.
    await screen.findByRole("heading", { level: 1 });

    // A flex-row container must sit below the header so the chat
    // column and ArtifactPanel can dock side-by-side on desktop.
    const splitRow = container.querySelector("div.flex-row");
    expect(splitRow).not.toBeNull();
    expect(splitRow?.className).toContain("flex-row");
    expect(
      screen.getByTestId("shared-chat-content-column").className,
    ).toContain("max-w-3xl");
  });

  test("auto-opens the most recent agent artifact in the shared chat", async () => {
    const oldArtifactId = "11111111-1111-4111-8111-111111111111";
    const latestArtifactId = "22222222-2222-4222-8222-222222222222";

    server.use(
      getGetV2GetSharedChatMockHandler200(
        (): SharedChatSession => ({
          id: "session-artifacts",
          title: "Artifacts",
          created_at: new Date("2026-05-12T00:00:00Z"),
          updated_at: new Date("2026-05-12T00:00:00Z"),
          shared_at: new Date("2026-05-12T00:00:00Z"),
          linked_executions: [],
        }),
      ),
      getGetV2GetSharedChatMessagesMockHandler200(
        (): SharedChatMessagesPage => ({
          messages: [
            {
              id: "m1",
              role: "assistant",
              content: `Older artifact [old.md](workspace://${oldArtifactId}#text/markdown)`,
              name: null,
              tool_call_id: null,
              tool_calls: null,
              function_call: null,
              sequence: 0,
              created_at: new Date("2026-05-12T00:00:01Z"),
            },
            {
              id: "m2",
              role: "assistant",
              content: `Latest artifact [latest.md](workspace://${latestArtifactId}#text/markdown)`,
              name: null,
              tool_call_id: null,
              tool_calls: null,
              function_call: null,
              sequence: 1,
              created_at: new Date("2026-05-12T00:00:02Z"),
            },
          ],
          has_more: false,
          oldest_sequence: 0,
        }),
      ),
      http.get(
        "*/api/proxy/api/public/shared/chats/:token/files/:fileId/download",
        () => new HttpResponse("# artifact"),
      ),
    );

    render(<SharedChatPage />);

    await screen.findByText(/Latest artifact/);
    await waitFor(() => {
      // The preview drawer is driven by activeArtifact, not `isOpen`.
      const panel = useCopilotUIStore.getState().artifactPanel;
      expect(panel.activeArtifact?.id).toBe(latestArtifactId);
      expect(panel.activeArtifact?.title).toBe("latest.md");
    });
    expect(
      screen.getByTestId("shared-chat-content-column").className,
    ).not.toContain("max-w-3xl");
  });

  test("falls back to 'Shared chat' when the session has no title", async () => {
    server.use(
      getGetV2GetSharedChatMockHandler200(
        (): SharedChatSession => ({
          id: "session-no-title",
          title: "",
          created_at: new Date("2026-05-12T00:00:00Z"),
          updated_at: new Date("2026-05-12T00:00:00Z"),
          shared_at: new Date("2026-05-12T00:00:00Z"),
          linked_executions: [],
        }),
      ),
      getGetV2GetSharedChatMessagesMockHandler200(
        (): SharedChatMessagesPage => ({
          messages: [],
          has_more: false,
          oldest_sequence: null,
        }),
      ),
    );

    render(<SharedChatPage />);
    const heading = await screen.findByRole("heading", { level: 1 });
    expect(heading.textContent).toBe("Shared chat");
  });

  test("does NOT show the has_more notice when the chat fits in one page", async () => {
    server.use(
      getGetV2GetSharedChatMockHandler200(
        (): SharedChatSession => ({
          id: "session-fits",
          title: "Fits",
          created_at: new Date("2026-05-12T00:00:00Z"),
          updated_at: new Date("2026-05-12T00:00:00Z"),
          shared_at: new Date("2026-05-12T00:00:00Z"),
          linked_executions: [],
        }),
      ),
      getGetV2GetSharedChatMessagesMockHandler200(
        (): SharedChatMessagesPage => ({
          messages: [],
          has_more: false,
          oldest_sequence: null,
        }),
      ),
    );

    render(<SharedChatPage />);
    await screen.findByRole("heading", { level: 1 });
    expect(screen.queryByText(/older history is not visible/i)).toBeNull();
  });

  describe("Cookie settings", () => {
    afterEach(() => {
      removeCookiebot();
      vi.unstubAllEnvs();
    });

    test("opens the Cookiebot dialog from the page footer", async () => {
      configureCookiebot();
      const { renew } = installCookiebot();
      server.use(
        http.get(
          "*/api/public/shared/chats/:token",
          () => new HttpResponse(null, { status: 404 }),
        ),
      );

      render(<SharedChatPage />);

      fireEvent.click(
        await screen.findByRole("button", { name: "Cookie settings" }),
      );

      expect(renew).toHaveBeenCalledOnce();
    });
  });
});
