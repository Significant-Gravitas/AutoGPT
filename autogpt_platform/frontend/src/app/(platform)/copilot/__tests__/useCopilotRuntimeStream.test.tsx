import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { useCopilotUIStore } from "../store";
import {
  getTurnRuntime,
  resetTurnRuntimes,
  SendInFlightError,
} from "../stream/turnRuntime";
import { useCopilotRuntimeStream } from "../useCopilotRuntimeStream";

const SESSION_ID = "session-refused-send";
const FIRST = "Draft this week's report";
const SECOND = "And the monthly one";

function resetStores() {
  resetTurnRuntimes();
  useCopilotStreamStore.getState().resetAll();
  useCopilotUIStore.setState({ initialPrompt: null });
}

beforeEach(resetStores);
afterEach(resetStores);

/** The session's runtime, whose first send waits for `refuse` to answer it. */
function holdFirstSend() {
  const sent: string[] = [];
  let answer: (response: Response) => void = () => {};
  getTurnRuntime(SESSION_ID, {
    baseUrl: () => "http://backend.test",
    headers: async () => ({}),
    fetch: ((_input: RequestInfo | URL, init?: RequestInit) => {
      sent.push(
        (JSON.parse(String(init?.body)) as { message: string }).message,
      );
      return new Promise<Response>((resolve) => {
        answer = resolve;
      });
    }) as typeof fetch,
    fetchSession: async () => null,
    toast: vi.fn(),
    report: vi.fn(),
  });
  return {
    sent,
    // The backend refuses a turn over the user's cap before saving its prompt.
    refuse: () =>
      answer(
        new Response(
          JSON.stringify({
            detail: "You've reached your usage limit. Try again later.",
          }),
          { status: 429, headers: { "Content-Type": "application/json" } },
        ),
      ),
  };
}

function renderStream() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return renderHook(
    () =>
      useCopilotRuntimeStream({
        sessionId: SESSION_ID,
        sessionView: null,
        hydratedMessages: [],
        refetchSession: async () => ({ data: undefined }),
        copilotModel: undefined,
      }),
    {
      wrapper: ({ children }: { children: ReactNode }) => (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      ),
    },
  );
}

it("puts back the text of a refused send, even after a second send was refused while it was out", async () => {
  const backend = holdFirstSend();
  const { result } = renderStream();

  let first: Promise<void> = Promise.resolve();
  let second: Promise<void> = Promise.resolve();
  act(() => {
    first = result.current.sendMessage({ text: FIRST });
    second = result.current.sendMessage({ text: SECOND });
  });
  await expect(second).rejects.toBeInstanceOf(SendInFlightError);
  await waitFor(() => expect(backend.sent).toEqual([FIRST]));

  await act(async () => {
    backend.refuse();
    await first;
  });

  // The composer gets back the text of the send the backend refused, not the
  // one the chat refused while that send was out.
  await waitFor(() =>
    expect(useCopilotUIStore.getState().initialPrompt).toBe(FIRST),
  );
});
