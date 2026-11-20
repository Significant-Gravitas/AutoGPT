import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SubSessionCard } from "../AgentCards";

const SESSION_ROUTE = "/api/proxy/api/chat/sessions/:sessionId";
/** Mirrors POLL_CAP_MS in SubSessionLive.tsx: where a foreground card gives up. */
const POLL_CAP_MS = 5 * 60_000;

function idleSession(outcome?: string) {
  return {
    id: "sub-1",
    chat_status: "idle",
    active_stream: null,
    background_outcome: outcome ?? null,
    messages: [
      {
        role: "user",
        content: "Work",
        metadata: { background_task_id: "task" },
      },
      { role: "assistant", content: "Partial result before recovery." },
    ],
  };
}

describe("background subtask recovery status", () => {
  afterEach(cleanup);

  it.each([false, true])(
    "keeps polling an idle child until recovery records its outcome (minimal=%s)",
    async (minimal) => {
      let requests = 0;
      server.use(
        http.get(SESSION_ROUTE, () =>
          HttpResponse.json(idleSession(++requests > 1 ? "failed" : undefined)),
        ),
      );
      render(
        <SubSessionCard
          minimal={minimal}
          output={{
            status: "queued",
            sub_session_id: "sub-1",
            background_task_id: "task",
          }}
        />,
      );
      if (!minimal) {
        await screen.findByText("Partial result before recovery.");
        expect(screen.queryByText(/^completed$/i)).toBeNull();
        expect(
          screen.getByRole("button", { name: "Stop background subtask" }),
        ).toBeDefined();
      }
      expect(
        await screen.findByText(/^failed$/i, {}, { timeout: 5000 }),
      ).toBeDefined();
      expect(
        screen.queryByRole("button", { name: "Stop background subtask" }),
      ).toBeNull();
    },
  );

  it("keeps following an unresolved idle child past the foreground poll cap", async () => {
    server.use(http.get(SESSION_ROUTE, () => HttpResponse.json(idleSession())));
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      render(
        <SubSessionCard
          output={{
            status: "queued",
            sub_session_id: "sub-1",
            background_task_id: "task",
          }}
        />,
      );
      await screen.findByText("Partial result before recovery.");
      await vi.advanceTimersByTimeAsync(POLL_CAP_MS + 1);
      expect(screen.queryByText("Live updates paused")).toBeNull();
      expect(screen.queryByText(/^unknown$/i)).toBeNull();
      expect(screen.queryByText(/^completed$/i)).toBeNull();
      expect(
        screen.getByRole("button", { name: "Stop background subtask" }),
      ).toBeDefined();
    } finally {
      vi.useRealTimers();
    }
  });
});
