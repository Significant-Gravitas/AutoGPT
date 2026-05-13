import { beforeEach, describe, expect, test, vi } from "vitest";
import { http, HttpResponse } from "msw";

import {
  getGetV2GetSharedChatMockHandler200,
  getGetV2GetSharedChatMessagesMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { SharedChatMessagesPage } from "@/app/api/__generated__/models/sharedChatMessagesPage";
import type { SharedChatSession } from "@/app/api/__generated__/models/sharedChatSession";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
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

beforeEach(() => {
  vi.clearAllMocks();
  mockUseParams.mockReturnValue({ token: TOKEN });
});

describe("SharedChatPage", () => {
  test("renders the read-only viewer with messages on happy path", async () => {
    server.use(
      getGetV2GetSharedChatMockHandler200(
        (): SharedChatSession => ({
          id: "session-123",
          title: "How to deploy",
          created_at: new Date("2026-05-12T00:00:00Z"),
          updated_at: new Date("2026-05-12T00:00:00Z"),
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

    expect(await screen.findByText("How to deploy")).toBeDefined();
    // Read-only banner is always visible on the happy path.
    expect(await screen.findByText(/public read-only view/i)).toBeDefined();
    expect(await screen.findByText("How do I deploy?")).toBeDefined();
    expect(await screen.findByText("Use docker compose.")).toBeDefined();
  });

  test("surfaces the has_more notice when the chat is truncated", async () => {
    server.use(
      getGetV2GetSharedChatMockHandler200(
        (): SharedChatSession => ({
          id: "session-123",
          title: "Long chat",
          created_at: new Date("2026-05-12T00:00:00Z"),
          updated_at: new Date("2026-05-12T00:00:00Z"),
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

  test("renders the not-found card when the share token is revoked", async () => {
    // Backend returns 404 for unknown/revoked tokens.
    server.use(
      http.get(
        "*/api/public/shared/chats/:token",
        () => new HttpResponse(null, { status: 404 }),
      ),
    );

    render(<SharedChatPage />);

    expect(await screen.findByText(/share link not found/i)).toBeDefined();
    expect(
      await screen.findByText(/invalid or has been disabled/i),
    ).toBeDefined();
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
    await screen.findByText("Layout check");

    const rootRow = container.querySelector("div.flex-row");
    expect(rootRow).not.toBeNull();
    expect(rootRow?.className).toContain("flex-row");
    expect(rootRow?.className).toContain("h-screen");
  });

  test("falls back to 'Shared chat' when the session has no title", async () => {
    server.use(
      getGetV2GetSharedChatMockHandler200(
        (): SharedChatSession => ({
          id: "session-no-title",
          title: "",
          created_at: new Date("2026-05-12T00:00:00Z"),
          updated_at: new Date("2026-05-12T00:00:00Z"),
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
    expect(await screen.findByText("Shared chat")).toBeDefined();
  });

  test("renders the persistent read-only indicators (subline + pill)", async () => {
    // The viewer must always tell visitors they're in a public,
    // read-only context.  We surface that twice in the persistent
    // header: a 'public read-only view' subline and a 'Read-only'
    // pill — neither should disappear if a chat is long.
    server.use(
      getGetV2GetSharedChatMockHandler200(
        (): SharedChatSession => ({
          id: "session-banner",
          title: "Banner check",
          created_at: new Date("2026-05-12T00:00:00Z"),
          updated_at: new Date("2026-05-12T00:00:00Z"),
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
    await screen.findByText("Banner check");
    expect(screen.getByText(/public read-only view/i)).toBeDefined();
    expect(screen.getByText("Read-only")).toBeDefined();
  });

  test("pins the 'Powered by AutoGPT Platform' footer outside the scroll area", async () => {
    // Regression for: a long chat could push the footer off-screen
    // or scroll over it.  The footer must live as a sibling of the
    // scrolling chat column so it stays visible regardless of message
    // count.
    server.use(
      getGetV2GetSharedChatMockHandler200(
        (): SharedChatSession => ({
          id: "session-footer",
          title: "Footer check",
          created_at: new Date("2026-05-12T00:00:00Z"),
          updated_at: new Date("2026-05-12T00:00:00Z"),
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
    await screen.findByText("Footer check");

    const footer = container.querySelector("footer");
    expect(footer).not.toBeNull();
    expect(footer?.textContent).toContain("Powered by AutoGPT Platform");
    // Footer is a shrink-0 sibling — never inside the scroll container.
    expect(footer?.className).toContain("shrink-0");
  });

  test("does NOT show the has_more notice when the chat fits in one page", async () => {
    server.use(
      getGetV2GetSharedChatMockHandler200(
        (): SharedChatSession => ({
          id: "session-fits",
          title: "Fits",
          created_at: new Date("2026-05-12T00:00:00Z"),
          updated_at: new Date("2026-05-12T00:00:00Z"),
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
    await screen.findByText("Fits");
    expect(screen.queryByText(/older history is not visible/i)).toBeNull();
  });
});
