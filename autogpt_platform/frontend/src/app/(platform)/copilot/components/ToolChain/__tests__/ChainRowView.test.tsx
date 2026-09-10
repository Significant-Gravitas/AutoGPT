import { getGetV2GetSessionMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it } from "vitest";
import type { ChainRow } from "../helpers";
import { ChainRowView } from "../ChainRowView";

function subSession(overrides: Record<string, unknown> = {}) {
  return getGetV2GetSessionMockHandler200({
    id: "sub-1",
    created_at: "2026-08-21T00:00:00Z",
    updated_at: "2026-08-21T00:00:00Z",
    user_id: "u-1",
    messages: [],
    ...overrides,
  });
}

function row(overrides: Partial<ChainRow>): ChainRow {
  return {
    key: "row-1",
    category: "agent",
    text: "Handing off to a teammate…",
    state: "running",
    ...overrides,
  };
}

afterEach(() => {
  server.resetHandlers();
});

describe("ChainRowView - live sub-session rows", () => {
  it("auto-opens and shows the pending card for a blocking delegate with no output yet", async () => {
    render(
      <ChainRowView
        row={row({
          tool: "delegate_to_expert",
          state: "running",
          output: undefined,
          input: { expert_id: "exp-1", prompt: "Create a chat app" },
        })}
        isLast
      />,
    );

    expect(await screen.findByText("Sub-AutoPilot")).toBeDefined();
    // Delegated cards are status-only — the prompt stays in the teammate's
    // own thread, not in the parent chain.
    expect(screen.queryByText("Create a chat app")).toBeNull();
  });

  it("does not auto-open a result-poll row with no output (nothing to show)", () => {
    render(
      <ChainRowView
        row={row({
          tool: "get_sub_session_result",
          state: "running",
          output: undefined,
          input: {},
        })}
        isLast
      />,
    );

    expect(screen.queryByText("Sub-AutoPilot")).toBeNull();
  });

  it("does not treat a non-sub-session tool as a live sub-session row", () => {
    render(
      <ChainRowView
        row={row({
          tool: "run_agent",
          state: "running",
          output: undefined,
          input: {},
        })}
        isLast
      />,
    );

    expect(screen.queryByText("Sub-AutoPilot")).toBeNull();
  });

  it("keeps showing the running label while a done delegate output is still working", async () => {
    server.use(subSession({ chat_status: "running", active_stream: false }));

    render(
      <ChainRowView
        row={row({
          tool: "delegate_to_expert",
          state: "done",
          output: { status: "running", sub_session_id: "sub-1" },
          input: { prompt: "Create a chat app" },
        })}
        isLast
      />,
    );

    expect(
      await screen.findByText(
        'Handing off to a teammate: "Create a chat app"…',
      ),
    ).toBeDefined();
  });

  it("does not claim the teammate finished when the poll dies", async () => {
    server.use(
      http.get("*/api/chat/sessions/:sessionId", () => {
        return new HttpResponse(null, { status: 500 });
      }),
    );

    render(
      <ChainRowView
        row={row({
          tool: "delegate_to_expert",
          state: "done",
          text: 'Teammate handled: "Create a chat app"',
          output: { status: "running", sub_session_id: "sub-1" },
          input: { prompt: "Create a chat app" },
        })}
        isLast
      />,
    );

    // A dead poll can't refute the frozen "running", so the done label must
    // never land — the row keeps saying the teammate is on it.
    await expect(
      screen.findByText('Teammate handled: "Create a chat app"'),
    ).rejects.toThrow();
    expect(
      screen.getByText('Handing off to a teammate: "Create a chat app"…'),
    ).toBeDefined();
  });

  it("switches to the done label once the polled sub-session goes idle", async () => {
    server.use(subSession({ chat_status: "completed" }));

    render(
      <ChainRowView
        row={row({
          tool: "delegate_to_expert",
          state: "done",
          text: 'Teammate handled: "Create a chat app"',
          output: { status: "running", sub_session_id: "sub-1" },
          input: { prompt: "Create a chat app" },
        })}
        isLast
      />,
    );

    expect(
      await screen.findByText('Teammate handled: "Create a chat app"'),
    ).toBeDefined();
  });
});
