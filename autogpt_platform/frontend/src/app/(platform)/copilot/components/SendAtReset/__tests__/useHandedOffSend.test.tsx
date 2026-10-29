import {
  getGetV2ListTurnsAtBudgetResetQueryKey,
  type getV2ListTurnsAtBudgetResetResponse,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { StrictMode, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useChatDraft } from "../../ChatInput/useChatDraft";
import { SEND_SETTLE_MS, sessionDraftKey } from "../helpers";
import { useScheduledSendHandoff } from "../scheduledSendHandoff";
import { useHandedOffSend } from "../useHandedOffSend";
import { useScheduledSend } from "../useScheduledSend";

vi.mock("@/lib/auth/hooks/useAuthStore", () => ({
  useAuthStore: (selector: (state: { user: { id: string } }) => unknown) =>
    selector({ user: { id: "user-a" } }),
}));

const toast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast,
  useToast: () => ({ toast, dismiss: vi.fn() }),
}));

const RUN_AT = new Date(2026, 9, 17, 1, 7).toISOString();
const MESSAGE = "Draft the launch post";

function withQueryClient(
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }),
) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <StrictMode>
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      </StrictMode>
    );
  };
}

function useNewChat(
  sessionId: string,
  onSendNow?: (message: string) => Promise<void>,
) {
  const scheduled = useScheduledSend({ sessionId, onSendNow });
  useHandedOffSend(sessionId, scheduled);
  const [draft] = useChatDraft(sessionDraftKey(sessionId));
  return {
    draft,
    pending: scheduled.pending,
    schedule: scheduled.schedule,
    cancel: scheduled.cancel,
    edit: scheduled.edit,
  };
}

function serverHolding(waiting: unknown[] | "unavailable") {
  const calls = { posted: [] as unknown[] };
  server.use(
    http.get("*/api/chat/sessions/:sessionId/at-reset", () =>
      waiting === "unavailable"
        ? HttpResponse.json(
            { detail: "Scheduler unavailable" },
            { status: 503 },
          )
        : HttpResponse.json({ jobs: waiting }),
    ),
    http.post(
      "*/api/chat/sessions/:sessionId/at-reset",
      async ({ request }) => {
        const body = (await request.json()) as { message: string };
        calls.posted.push(body);
        return HttpResponse.json({
          kind: "send",
          run_at: RUN_AT,
          message: body.message,
        });
      },
    ),
  );
  return calls;
}

function handedOff(sessionId: string) {
  return useScheduledSendHandoff.getState().sends[sessionId];
}

function toastTitles() {
  return toast.mock.calls.map(([options]) => options.title);
}

beforeEach(() => {
  sessionStorage.clear();
  useScheduledSendHandoff.setState({ sends: {} });
  toast.mockClear();
});

describe("useHandedOffSend", () => {
  it("sets a send the new-chat screen handed over once, then lets it go", async () => {
    const calls = serverHolding([]);
    useScheduledSendHandoff.getState().hold("chat-new", MESSAGE);

    renderHook(() => useNewChat("chat-new"), { wrapper: withQueryClient() });

    await waitFor(() => expect(handedOff("chat-new")).toBeUndefined());
    expect(calls.posted).toEqual([{ kind: "send", message: MESSAGE }]);
  });

  it("keeps it, marked sent, until the request setting it settles", async () => {
    let release = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    let posted = 0;
    server.use(
      http.get("*/api/chat/sessions/:sessionId/at-reset", () =>
        HttpResponse.json({ jobs: [] }),
      ),
      http.post("*/api/chat/sessions/:sessionId/at-reset", async () => {
        posted += 1;
        await held;
        return HttpResponse.json({
          kind: "send",
          run_at: RUN_AT,
          message: MESSAGE,
        });
      }),
    );
    useScheduledSendHandoff.getState().hold("chat-in-flight", MESSAGE);

    renderHook(() => useNewChat("chat-in-flight"), {
      wrapper: withQueryClient(),
    });

    await waitFor(() => expect(posted).toBe(1));
    // What a reload now would find.
    expect(handedOff("chat-in-flight")).toEqual({
      message: MESSAGE,
      sentAt: expect.any(Number),
    });
    release();
    await waitFor(() => expect(handedOff("chat-in-flight")).toBeUndefined());
  });

  // The budget came back meanwhile, so the message goes out now and its reply
  // streams long after the server answered. A reload during the reply mustn't
  // find the message and put it back in the composer.
  it("lets it go once the server says to send it now", async () => {
    server.use(
      http.get("*/api/chat/sessions/:sessionId/at-reset", () =>
        HttpResponse.json({ jobs: [] }),
      ),
      http.post("*/api/chat/sessions/:sessionId/at-reset", () =>
        HttpResponse.json({ detail: "not_out_of_budget" }, { status: 409 }),
      ),
    );
    const sendNow = vi.fn(() => new Promise<void>(() => {}));
    useScheduledSendHandoff.getState().hold("chat-budget-back", MESSAGE);

    const { result } = renderHook(
      () => useNewChat("chat-budget-back", sendNow),
      { wrapper: withQueryClient() },
    );

    await waitFor(() => expect(sendNow).toHaveBeenCalledWith(MESSAGE));
    expect(handedOff("chat-budget-back")).toBeUndefined();
    expect(result.current.draft).toBe("");
    expect(toastTitles()).toEqual([]);
  });

  describe("after a reload while the send was being set", () => {
    // By default long enough ago that its request has landed if it ever will.
    function sentBeforeTheReload(sessionId: string, ago = SEND_SETTLE_MS) {
      useScheduledSendHandoff.setState({
        sends: { [sessionId]: { message: MESSAGE, sentAt: Date.now() - ago } },
      });
    }

    // The chat's list as the page read it on opening: no send, and fresh, so
    // only the page's own reads of the send reach the server.
    function listedEmpty(sessionId: string) {
      const client = new QueryClient({
        defaultOptions: { queries: { retry: false, staleTime: Infinity } },
      });
      client.setQueryData<getV2ListTurnsAtBudgetResetResponse>(
        getGetV2ListTurnsAtBudgetResetQueryKey(sessionId),
        { status: 200, data: { jobs: [] }, headers: new Headers() },
      );
      return client;
    }

    it("keeps it when the server has it, joined onto what was waiting", async () => {
      const calls = serverHolding([
        { kind: "send", run_at: RUN_AT, message: `Earlier note\n\n${MESSAGE}` },
      ]);
      sentBeforeTheReload("chat-landed");

      const { result } = renderHook(() => useNewChat("chat-landed"), {
        wrapper: withQueryClient(),
      });

      await waitFor(() => expect(handedOff("chat-landed")).toBeUndefined());
      expect(calls.posted).toEqual([]);
      expect(result.current.draft).toBe("");
      expect(toastTitles()).toEqual([]);
    });

    // The chat's list, read as the page opened, can predate the send landing,
    // and nothing reads a list holding no send again.
    it("shows the send it finds the server has", async () => {
      const client = listedEmpty("chat-listed-early");
      serverHolding([{ kind: "send", run_at: RUN_AT, message: MESSAGE }]);
      sentBeforeTheReload("chat-listed-early");

      const { result } = renderHook(() => useNewChat("chat-listed-early"), {
        wrapper: withQueryClient(client),
      });

      await waitFor(() =>
        expect(result.current.pending?.message).toBe(MESSAGE),
      );
      expect(handedOff("chat-listed-early")).toBeUndefined();
      expect(result.current.draft).toBe("");
    });

    // The server keeps on with a request after a reload drops the page that
    // sent it, so a send missing just after the reload can still land.
    it("waits for a request still landing before putting it back", async () => {
      // The clock stands still, so the page reads before the window ends
      // however long it takes to open; the wait itself runs in real time.
      vi.useFakeTimers({ toFake: ["Date"] });
      try {
        vi.setSystemTime(new Date(2026, 9, 18, 9, 0));
        const client = listedEmpty("chat-landing");
        let reads = 0;
        server.use(
          http.get("*/api/chat/sessions/:sessionId/at-reset", () => {
            reads += 1;
            return HttpResponse.json({
              jobs:
                reads > 1
                  ? [{ kind: "send", run_at: RUN_AT, message: MESSAGE }]
                  : [],
            });
          }),
        );
        sentBeforeTheReload("chat-landing", SEND_SETTLE_MS - 300);

        const { result } = renderHook(() => useNewChat("chat-landing"), {
          wrapper: withQueryClient(client),
        });

        await waitFor(
          () => expect(result.current.pending?.message).toBe(MESSAGE),
          { timeout: 3000 },
        );
        expect(handedOff("chat-landing")).toBeUndefined();
        expect(result.current.draft).toBe("");
        expect(toastTitles()).toEqual([]);
      } finally {
        vi.useRealTimers();
      }
    });

    describe("when the user changes the send during the wait", () => {
      beforeEach(() => {
        // The clock stands still, so the page reads before the window ends
        // however long it takes to open; the wait itself runs in real time.
        vi.useFakeTimers({ toFake: ["Date"] });
        vi.setSystemTime(new Date(2026, 9, 18, 9, 0));
      });
      afterEach(() => {
        vi.useRealTimers();
      });

      // The page has read the send once and waits for the gone page's
      // request, which lands right after that read if `lands`.
      async function waitingChat(sessionId: string, lands: boolean) {
        const calls = { reads: 0, cancelled: 0 };
        let waiting: string | null = null;
        server.use(
          http.get("*/api/chat/sessions/:sessionId/at-reset", () => {
            calls.reads += 1;
            const jobs = waiting
              ? [{ kind: "send", run_at: RUN_AT, message: waiting }]
              : [];
            if (lands && calls.reads === 1) waiting = MESSAGE;
            return HttpResponse.json({ jobs });
          }),
          http.post(
            "*/api/chat/sessions/:sessionId/at-reset",
            async ({ request }) => {
              const body = (await request.json()) as { message: string };
              waiting = waiting
                ? `${waiting}\n\n${body.message}`
                : body.message;
              return HttpResponse.json({
                kind: "send",
                run_at: RUN_AT,
                message: waiting,
              });
            },
          ),
          http.delete("*/api/chat/sessions/:sessionId/at-reset/:kind", () => {
            calls.cancelled += 1;
            waiting = null;
            return new HttpResponse(null, { status: 204 });
          }),
        );
        sentBeforeTheReload(sessionId, SEND_SETTLE_MS - 1000);
        const client = listedEmpty(sessionId);
        const { result } = renderHook(() => useNewChat(sessionId), {
          wrapper: withQueryClient(client),
        });
        await waitFor(() => expect(calls.reads).toBe(1));
        return { calls, client, result };
      }

      // The request landed, the chip showed it, and the user cancelled or
      // edited it: putting the message back would undo what they did.
      it.each(["cancel", "edit"] as const)(
        "leaves a send that landed to the user once they %s it",
        async (action) => {
          const sessionId = `chat-landed-${action}`;
          const { calls, client, result } = await waitingChat(sessionId, true);
          await act(() =>
            client.invalidateQueries({
              queryKey: getGetV2ListTurnsAtBudgetResetQueryKey(sessionId),
            }),
          );
          await waitFor(() =>
            expect(result.current.pending?.message).toBe(MESSAGE),
          );

          await act(async () => {
            await result.current[action]();
          });

          await waitFor(() => expect(handedOff(sessionId)).toBeUndefined(), {
            timeout: 3000,
          });
          expect(calls.cancelled).toBe(1);
          expect(result.current.draft).toBe(action === "edit" ? MESSAGE : "");
          expect(toastTitles()).toEqual([]);
        },
      );

      // A message the user sets meanwhile doesn't stand in for the one the
      // gone page's request was carrying, which never landed.
      it("still puts it back once the user adds a message of their own", async () => {
        const { result } = await waitingChat("chat-added-to", false);

        await act(() => result.current.schedule("Then the newsletter"));

        await waitFor(() => expect(result.current.draft).toBe(MESSAGE), {
          timeout: 3000,
        });
        expect(handedOff("chat-added-to")).toBeUndefined();
        expect(toastTitles()).toEqual(["Couldn’t schedule your message"]);
      });

      // Nor does cancelling or editing that message: it never held this one.
      it.each(["cancel", "edit"] as const)(
        "still puts it back once the user %ss a message of their own",
        async (action) => {
          const sessionId = `chat-own-${action}`;
          const { calls, result } = await waitingChat(sessionId, false);
          await act(() => result.current.schedule("Then the newsletter"));

          await act(async () => {
            await result.current[action]();
          });

          await waitFor(
            () =>
              expect(result.current.draft).toBe(
                action === "edit"
                  ? `${MESSAGE}\n\nThen the newsletter`
                  : MESSAGE,
              ),
            { timeout: 3000 },
          );
          expect(calls.cancelled).toBe(1);
          expect(handedOff(sessionId)).toBeUndefined();
          expect(toastTitles()).toEqual(["Couldn’t schedule your message"]);
        },
      );
    });

    it("puts it back in the composer when the server doesn't have it", async () => {
      const calls = serverHolding([]);
      sentBeforeTheReload("chat-lost");

      const { result } = renderHook(() => useNewChat("chat-lost"), {
        wrapper: withQueryClient(),
      });

      await waitFor(() => expect(result.current.draft).toBe(MESSAGE));
      expect(handedOff("chat-lost")).toBeUndefined();
      expect(calls.posted).toEqual([]);
      expect(toastTitles()).toEqual(["Couldn’t schedule your message"]);
    });

    it("puts it back in the composer when the server can't be asked", async () => {
      const calls = serverHolding("unavailable");
      sentBeforeTheReload("chat-unsure");

      const { result } = renderHook(() => useNewChat("chat-unsure"), {
        wrapper: withQueryClient(),
      });

      await waitFor(() => expect(result.current.draft).toBe(MESSAGE));
      expect(calls.posted).toEqual([]);
      expect(toastTitles()).toEqual(["Couldn’t check your scheduled message"]);
      // Sending it again could join it on twice, so the toast says why not.
      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({
          description: expect.stringContaining("It may already be set to send"),
        }),
      );
    });
  });
});
