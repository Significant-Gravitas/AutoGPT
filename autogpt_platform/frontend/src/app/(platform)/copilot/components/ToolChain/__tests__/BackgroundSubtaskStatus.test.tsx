import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it } from "vitest";
import { SubSessionCard } from "../AgentCards";

const SESSION_ROUTE = "/api/proxy/api/chat/sessions/:sessionId";

function idleSession(outcome?: string) {
  return {
    id: "sub-1",
    chat_status: "idle",
    active_stream: null,
    messages: [
      {
        role: "user",
        content: "Work",
        metadata: {
          background_task_id: "task",
          background_task_outcome: outcome,
        },
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
            background_expires_at: new Date(Date.now() + 60_000).toISOString(),
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

  it("pauses an unresolved idle child at the server expiry without claiming success", async () => {
    server.use(http.get(SESSION_ROUTE, () => HttpResponse.json(idleSession())));
    render(
      <SubSessionCard
        output={{
          status: "queued",
          sub_session_id: "sub-1",
          background_task_id: "task",
          background_expires_at: new Date(Date.now() - 1000).toISOString(),
        }}
      />,
    );
    await screen.findByText("Partial result before recovery.");
    expect(await screen.findByText("Live updates paused")).toBeDefined();
    expect(screen.getByText(/^unknown$/i)).toBeDefined();
    expect(screen.queryByText(/^completed$/i)).toBeNull();
  });
});
