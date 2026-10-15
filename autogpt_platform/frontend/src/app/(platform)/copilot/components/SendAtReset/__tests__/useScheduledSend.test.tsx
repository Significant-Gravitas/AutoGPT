import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { delay, http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { useChatDraft } from "../../ChatInput/useChatDraft";
import { sessionDraftKey } from "../helpers";
import { useScheduledSend } from "../useScheduledSend";

vi.mock("@/lib/auth/hooks/useAuthStore", () => ({
  useAuthStore: (selector: (state: { user: { id: string } }) => unknown) =>
    selector({ user: { id: "user-a" } }),
}));

function withQueryClient() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
}

describe("useScheduledSend", () => {
  // It runs in ChatContainer, which re-rendered on every keystroke while it
  // followed the draft it only ever writes.
  it("leaves the chat alone while its draft is typed", async () => {
    let reads = 0;
    server.use(
      http.get("*/api/chat/sessions/:sessionId/at-reset", () => {
        reads += 1;
        return HttpResponse.json({ jobs: [] });
      }),
    );
    let renders = 0;
    renderHook(
      () => {
        renders += 1;
        return useScheduledSend({ sessionId: "chat-typing" });
      },
      { wrapper: withQueryClient() },
    );
    await waitFor(() => expect(reads).toBe(1));
    await act(() => delay(100));
    const composer = renderHook(() =>
      useChatDraft(sessionDraftKey("chat-typing")),
    );
    const settled = renders;

    act(() => composer.result.current[1]("Draft the launch post"));
    act(() => composer.result.current[1]("Draft the launch post today"));

    expect(renders).toBe(settled);
  });
});
