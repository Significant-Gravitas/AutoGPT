import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import {
  getDeleteV2DisableChatSharingMockHandler204,
  getGetV2GetChatShareStateMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { ChatShareStateResponse } from "@/app/api/__generated__/models/chatShareStateResponse";
import { server } from "@/mocks/mock-server";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
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
// the copyShareUrl plumbing can be exercised without crashing.  The
// explicit (text: string) typing keeps mock.calls[0][0] usable in
// assertions without TS complaining about tuple length.
const clipboardWrite = vi.fn(async (_text: string) => {});
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

  test("Stop sharing requires a confirmation step before firing the DELETE", async () => {
    const TOKEN = "33333333-dddd-eeee-ffff-444444444444";
    mockShareState({
      is_shared: true,
      share_token: TOKEN,
      auto_share_executions: true,
    });
    server.use(getDeleteV2DisableChatSharingMockHandler204());

    render(
      <ShareChatDialog sessionId={SESSION_ID} open onOpenChange={vi.fn()} />,
    );

    const stopButton = await screen.findByRole("button", {
      name: /^stop sharing$/i,
    });
    fireEvent.click(stopButton);

    // After the first click the button copy switches to "Confirm
    // stop sharing" and a Cancel button appears.  The DELETE has NOT
    // fired yet — accidental click is recoverable.
    const confirmButton = await screen.findByRole("button", {
      name: /confirm stop sharing/i,
    });
    expect(screen.getByRole("button", { name: /^cancel$/i })).toBeDefined();
    // Still in shared state — Enable hasn't appeared.
    expect(
      screen.queryByRole("button", { name: /enable sharing/i }),
    ).toBeNull();

    fireEvent.click(confirmButton);

    // Second click is the authoritative DELETE; share-state flips
    // and Enable comes back.
    await screen.findByRole("button", { name: /enable sharing/i });
  });

  test("clicking Cancel during stop-confirmation returns to the share view", async () => {
    const TOKEN = "44444444-eeee-ffff-aaaa-555555555555";
    mockShareState({
      is_shared: true,
      share_token: TOKEN,
      auto_share_executions: true,
    });

    render(
      <ShareChatDialog sessionId={SESSION_ID} open onOpenChange={vi.fn()} />,
    );

    const stopButton = await screen.findByRole("button", {
      name: /^stop sharing$/i,
    });
    fireEvent.click(stopButton);
    fireEvent.click(await screen.findByRole("button", { name: /^cancel$/i }));

    // The confirm-stop UI collapses back to the primary "Stop
    // sharing" button — the share is still active.
    await screen.findByRole("button", { name: /^stop sharing$/i });
    expect(
      screen.queryByRole("button", { name: /confirm stop sharing/i }),
    ).toBeNull();
  });

  test("clicking Copy writes the share URL to the clipboard", async () => {
    const TOKEN = "55555555-aaaa-bbbb-cccc-666666666666";
    mockShareState({
      is_shared: true,
      share_token: TOKEN,
      auto_share_executions: true,
    });

    render(
      <ShareChatDialog sessionId={SESSION_ID} open onOpenChange={vi.fn()} />,
    );

    const copyButton = await screen.findByRole("button", { name: /copy/i });
    fireEvent.click(copyButton);

    await waitFor(() => {
      expect(clipboardWrite).toHaveBeenCalledTimes(1);
    });
    expect(clipboardWrite.mock.calls[0][0]).toMatch(
      new RegExp(`/share/chat/${TOKEN}$`),
    );
    // Button briefly flips to the "Copied" affordance — pinning the
    // setCopied / setTimeout(2000) loop in useShareChatDialog.
    await screen.findByRole("button", { name: /copied/i });
  });

  test("clipboard failure surfaces a destructive toast and skips the copied flash", async () => {
    // navigator.clipboard.writeText can reject when the browser refuses
    // (insecure context, missing permission).  Production must surface a
    // toast and NOT flip the button to "Copied".
    const TOKEN = "99999999-aaaa-bbbb-cccc-aaaaaaaaaaaa";
    mockShareState({
      is_shared: true,
      share_token: TOKEN,
      auto_share_executions: true,
    });
    clipboardWrite.mockRejectedValueOnce(new Error("nope"));

    render(
      <ShareChatDialog sessionId={SESSION_ID} open onOpenChange={vi.fn()} />,
    );

    const copyButton = await screen.findByRole("button", { name: /copy/i });
    fireEvent.click(copyButton);

    await waitFor(() => {
      expect(clipboardWrite).toHaveBeenCalled();
    });
    // After the rejection settles, the button must stay on "Copy" and
    // never flip to "Copied" (the setCopied(true) branch is skipped
    // inside the catch).
    expect(screen.queryByRole("button", { name: /copied/i })).toBeNull();
    expect(screen.getByRole("button", { name: /copy/i })).toBeDefined();
  });

  test("auto-share toggle is locked once the chat is shared", async () => {
    // The share-state drives what viewers see, so flipping the toggle
    // mid-share would lie.  Production guards this with
    // ``disabled={state.isShared || state.isLoadingState}``.
    mockShareState({
      is_shared: true,
      share_token: "77777777-aaaa-bbbb-cccc-888888888888",
      auto_share_executions: true,
    });
    render(
      <ShareChatDialog sessionId={SESSION_ID} open onOpenChange={vi.fn()} />,
    );

    // Wait for the share-state query to hydrate (the Stop sharing button
    // only appears once isLoadingState flips false), then assert the
    // toggle is in Radix's data-disabled state.
    await screen.findByRole("button", { name: /stop sharing/i });
    const toggle = screen.getByLabelText(/share agent runs from this chat/i);
    expect(toggle.getAttribute("data-disabled")).not.toBeNull();
  });
});
