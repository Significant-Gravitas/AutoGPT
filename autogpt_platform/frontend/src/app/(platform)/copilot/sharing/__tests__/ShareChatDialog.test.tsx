import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import { getGetV2GetChatShareStateMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { ChatShareStateResponse } from "@/app/api/__generated__/models/chatShareStateResponse";
import { server } from "@/mocks/mock-server";
import { cleanup, render, screen } from "@/tests/integrations/test-utils";
import { ShareChatDialog } from "../ShareChatDialog";

const SESSION_ID = "550e8400-e29b-41d4-a716-446655440000";

vi.mock("next/navigation", () => ({
  usePathname: () => "/copilot",
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

// Clipboard isn't available in jsdom by default; defineProperty so
// the copyShareUrl plumbing can be exercised without crashing.
const clipboardWrite = vi.fn(() => Promise.resolve());
beforeEach(() => {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: clipboardWrite },
  });
  clipboardWrite.mockClear();
});

afterEach(() => {
  cleanup();
});

function mockShareState(state: Partial<ChatShareStateResponse>) {
  server.use(
    getGetV2GetChatShareStateMockHandler200(
      (): ChatShareStateResponse => ({
        is_shared: false,
        share_token: null,
        auto_share_executions: false,
        ...state,
      }),
    ),
  );
}

describe("ShareChatDialog", () => {
  test("opens in 'Enable sharing' mode for an unshared chat", async () => {
    mockShareState({ is_shared: false });
    render(
      <ShareChatDialog sessionId={SESSION_ID} open onOpenChange={vi.fn()} />,
    );

    expect(
      await screen.findByRole("button", { name: /enable sharing/i }),
    ).toBeDefined();
    // Read-only warning always visible.
    expect(
      screen.getByText(/anyone with the link will see this conversation/i),
    ).toBeDefined();
    // The auto-share toggle is present and labelled — the always-show
    // box the user asked for, regardless of run count.
    expect(
      screen.getByLabelText(/share agent runs from this chat/i),
    ).toBeDefined();
  });

  test("opens in 'Stop sharing' mode and shows the share URL for an already-shared chat", async () => {
    const TOKEN = "11111111-2222-3333-4444-555555555555";
    mockShareState({
      is_shared: true,
      share_token: TOKEN,
      auto_share_executions: true,
    });
    render(
      <ShareChatDialog sessionId={SESSION_ID} open onOpenChange={vi.fn()} />,
    );

    expect(
      await screen.findByRole("button", { name: /stop sharing/i }),
    ).toBeDefined();
    // Share URL surfaces inside the dialog so the owner can copy it.
    const urlInput = await screen.findByDisplayValue(
      new RegExp(`/share/chat/${TOKEN}$`),
    );
    expect(urlInput).toBeDefined();
    expect(screen.getByRole("button", { name: /copy/i })).toBeDefined();
  });

  test("renders the auto-share toggle when chat is already shared", async () => {
    mockShareState({
      is_shared: true,
      share_token: "11111111-2222-3333-4444-555555555555",
      auto_share_executions: true,
    });
    render(
      <ShareChatDialog sessionId={SESSION_ID} open onOpenChange={vi.fn()} />,
    );

    // Wait for the share-state query to resolve and hydrate the dialog.
    await screen.findByRole("button", { name: /stop sharing/i });

    // The toggle is still rendered (always-visible by design) and
    // labelled — its visual checked-state is owned by the Switch
    // component implementation; we just pin that it exists and is
    // reachable via its accessible label.
    expect(
      screen.getByLabelText(/share agent runs from this chat/i),
    ).toBeDefined();
  });

  test("does not render dialog content when open=false", () => {
    mockShareState({ is_shared: false });
    render(
      <ShareChatDialog
        sessionId={SESSION_ID}
        open={false}
        onOpenChange={vi.fn()}
      />,
    );
    expect(
      screen.queryByRole("button", { name: /enable sharing/i }),
    ).toBeNull();
    expect(screen.queryByRole("button", { name: /stop sharing/i })).toBeNull();
  });
});
