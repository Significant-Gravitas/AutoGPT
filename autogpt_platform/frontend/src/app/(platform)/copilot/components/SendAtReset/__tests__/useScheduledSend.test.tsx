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

  describe("Edit", () => {
    const RUN_AT = new Date(2026, 9, 17, 1, 7).toISOString();

    function serverHolding(jobs: () => unknown[]) {
      const calls = { cancelled: 0 };
      server.use(
        http.get("*/api/chat/sessions/:sessionId/at-reset", () =>
          HttpResponse.json({ jobs: jobs() }),
        ),
        http.delete("*/api/chat/sessions/:sessionId/at-reset/:kind", () => {
          calls.cancelled += 1;
          return new HttpResponse(null, { status: 204 });
        }),
      );
      return calls;
    }

    // Another tab or client can add to the waiting send after this one last
    // read the list.
    it("puts back the send as the server holds it now", async () => {
      let message = "Draft the launch post";
      const calls = serverHolding(() => [
        { kind: "send", run_at: RUN_AT, message },
      ]);
      const { result } = renderHook(
        () => useScheduledSend({ sessionId: "chat-edit" }),
        { wrapper: withQueryClient() },
      );
      await waitFor(() =>
        expect(result.current.pending?.message).toBe("Draft the launch post"),
      );
      message = "Draft the launch post\n\nThen the newsletter";
      const composer = renderHook(() =>
        useChatDraft(sessionDraftKey("chat-edit")),
      );

      await act(() => result.current.edit());

      expect(calls.cancelled).toBe(1);
      expect(composer.result.current[0]).toBe(
        "Draft the launch post\n\nThen the newsletter",
      );
    });

    it("leaves a send that has already gone out of the draft", async () => {
      let jobs: unknown[] = [
        { kind: "send", run_at: RUN_AT, message: "Draft the launch post" },
      ];
      const calls = serverHolding(() => jobs);
      const { result } = renderHook(
        () => useScheduledSend({ sessionId: "chat-sent" }),
        { wrapper: withQueryClient() },
      );
      await waitFor(() => expect(result.current.pending).not.toBeNull());
      jobs = [];
      const composer = renderHook(() =>
        useChatDraft(sessionDraftKey("chat-sent")),
      );

      await act(() => result.current.edit());

      expect(calls.cancelled).toBe(0);
      expect(composer.result.current[0]).toBe("");
      await waitFor(() => expect(result.current.pending).toBeNull());
    });

    // Cancelling on the copy this tab last read could drop text the server
    // holds, so a failed read changes nothing.
    it("changes nothing when it can't read the server's copy", async () => {
      let reads = 0;
      let cancelled = 0;
      server.use(
        http.get("*/api/chat/sessions/:sessionId/at-reset", () => {
          reads += 1;
          return reads === 1
            ? HttpResponse.json({
                jobs: [
                  {
                    kind: "send",
                    run_at: RUN_AT,
                    message: "Draft the launch post",
                  },
                ],
              })
            : HttpResponse.json(
                { detail: "Scheduler unavailable" },
                { status: 503 },
              );
        }),
        http.delete("*/api/chat/sessions/:sessionId/at-reset/:kind", () => {
          cancelled += 1;
          return new HttpResponse(null, { status: 204 });
        }),
      );
      const { result } = renderHook(
        () => useScheduledSend({ sessionId: "chat-unread" }),
        { wrapper: withQueryClient() },
      );
      await waitFor(() => expect(result.current.pending).not.toBeNull());
      const composer = renderHook(() =>
        useChatDraft(sessionDraftKey("chat-unread")),
      );

      await act(() => result.current.edit());

      expect(reads).toBe(2);
      expect(cancelled).toBe(0);
      expect(composer.result.current[0]).toBe("");
    });

    // The composer's Edit and the "send at reset" menu each change the same
    // chat's send. A message the menu adds while Edit reads may not be in
    // what Edit read, and cancelling that copy would drop the message.
    describe("with a message added while it reads", () => {
      const FIRST = "Draft the launch post";
      const JOINED = `${FIRST}\n\nThen the newsletter`;

      // The server holds the first message, then both once the add lands. A
      // read answers with what the server held when the read came in.
      function serverJoining() {
        const calls = { reads: 0, added: 0, cancelled: 0 };
        const hold: { nextRead?: Promise<void>; add?: Promise<void> } = {};
        let waiting: string | null = FIRST;
        server.use(
          http.get("*/api/chat/sessions/:sessionId/at-reset", async () => {
            calls.reads += 1;
            const answer = waiting;
            const held = hold.nextRead;
            hold.nextRead = undefined;
            await held;
            return HttpResponse.json({
              jobs: answer
                ? [{ kind: "send", run_at: RUN_AT, message: answer }]
                : [],
            });
          }),
          http.post("*/api/chat/sessions/:sessionId/at-reset", async () => {
            calls.added += 1;
            await hold.add;
            waiting = JOINED;
            return HttpResponse.json({
              kind: "send",
              run_at: RUN_AT,
              message: JOINED,
            });
          }),
          http.delete("*/api/chat/sessions/:sessionId/at-reset/:kind", () => {
            calls.cancelled += 1;
            waiting = null;
            return new HttpResponse(null, { status: 204 });
          }),
        );
        return { calls, hold };
      }

      async function openChat(sessionId: string) {
        const client = new QueryClient({
          defaultOptions: { queries: { retry: false, staleTime: Infinity } },
        });
        function wrapper({ children }: { children: ReactNode }) {
          return (
            <QueryClientProvider client={client}>
              {children}
            </QueryClientProvider>
          );
        }
        const chat = {
          composer: renderHook(() => useScheduledSend({ sessionId }), {
            wrapper,
          }),
          menu: renderHook(() => useScheduledSend({ sessionId }), { wrapper }),
          draft: renderHook(() => useChatDraft(sessionDraftKey(sessionId))),
        };
        await waitFor(() =>
          expect(chat.composer.result.current.pending?.message).toBe(FIRST),
        );
        return chat;
      }

      function gate() {
        let release = () => {};
        const closed = new Promise<void>((resolve) => {
          release = resolve;
        });
        return { closed, release };
      }

      it("leaves it alone when the message lands before the read", async () => {
        const { calls, hold } = serverJoining();
        const chat = await openChat("chat-added-during-read");
        const read = gate();
        hold.nextRead = read.closed;
        const readsBefore = calls.reads;

        let editing = Promise.resolve();
        act(() => {
          editing = chat.composer.result.current.edit();
        });
        await waitFor(() => expect(calls.reads).toBe(readsBefore + 1));
        await act(() =>
          chat.menu.result.current.schedule("Then the newsletter"),
        );
        read.release();
        await act(() => editing);

        expect(calls.cancelled).toBe(0);
        expect(chat.draft.result.current[0]).toBe("");
        await waitFor(() =>
          expect(chat.composer.result.current.pending?.message).toBe(JOINED),
        );
      });

      it("still edits once a change it made has settled", async () => {
        const { calls } = serverJoining();
        const chat = await openChat("chat-edit-after-add");
        await act(() =>
          chat.menu.result.current.schedule("Then the newsletter"),
        );

        await act(() => chat.composer.result.current.edit());

        expect(calls.cancelled).toBe(1);
        expect(chat.draft.result.current[0]).toBe(JOINED);
      });

      it("leaves it alone when the read lands before the message", async () => {
        const { calls, hold } = serverJoining();
        const chat = await openChat("chat-read-during-add");
        const add = gate();
        hold.add = add.closed;

        let adding = Promise.resolve();
        act(() => {
          adding = chat.menu.result.current.schedule("Then the newsletter");
        });
        await waitFor(() => expect(calls.added).toBe(1));
        await act(() => chat.composer.result.current.edit());
        add.release();
        await act(() => adding);

        expect(calls.cancelled).toBe(0);
        expect(chat.draft.result.current[0]).toBe("");
        await waitFor(() =>
          expect(chat.composer.result.current.pending?.message).toBe(JOINED),
        );
      });
    });
  });
});
