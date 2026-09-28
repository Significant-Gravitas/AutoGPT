import { getGetV2GetSessionMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SubSessionCard } from "../AgentCards";
import { SubSessionPendingCard } from "../SubSessionLive";
import { ToolResult } from "../ToolResult";

/** Mirrors POLL_CAP_MS in SubSessionLive.tsx — the poll gives up after this.
 *  Kept local so the test states the contract rather than importing it. */
const POLL_CAP_MS = 5 * 60_000;

const SESSION_ROUTE = "/api/proxy/api/chat/sessions/:sessionId";
const SESSION_LIST_ROUTE = "/api/proxy/api/chat/sessions";

function failingSubSession() {
  return http.get(SESSION_ROUTE, () => new HttpResponse(null, { status: 500 }));
}

function summary(id: string, extra: Record<string, unknown>) {
  return {
    id,
    created_at: "2026-08-21T00:00:00Z",
    updated_at: "2026-08-21T00:00:00Z",
    title: id,
    chat_status: "idle",
    is_processing: false,
    is_pinned: false,
    expert_id: "exp-1",
    ...extra,
  };
}

/** Mimics the backend default: pinned sessions win over recency, so a running
 *  session only reaches a small page when pinned_first is explicitly off. */
function expertSessions() {
  return http.get(SESSION_LIST_ROUTE, ({ request }) => {
    const pinnedFirst =
      new URL(request.url).searchParams.get("pinned_first") !== "false";
    const pinned = Array.from({ length: 5 }, (_, i) =>
      summary(`pin-${i}`, { is_pinned: true }),
    );
    const running = summary("sub-live", {
      chat_status: "running",
      is_processing: true,
    });
    return HttpResponse.json({
      sessions: pinnedFirst ? pinned : [running, ...pinned],
      total: 6,
    });
  });
}

function subSession(
  messages: Record<string, unknown>[],
  extra: Record<string, unknown> = {},
) {
  return getGetV2GetSessionMockHandler200({
    id: "sub-1",
    created_at: "2026-08-21T00:00:00Z",
    updated_at: "2026-08-21T00:00:00Z",
    user_id: "u-1",
    messages,
    ...extra,
  });
}

describe("SubSessionLive", () => {
  afterEach(cleanup);

  it.each([
    ["run_agent", "Daily briefing", "Running agent", "Daily briefing"],
    [
      "run_block",
      "FillTextTemplateBlock",
      "Running block",
      "Fill Text Template",
    ],
    [
      "continue_run_block",
      "FillTextTemplateBlock",
      "Continuing block run",
      "Fill Text Template",
    ],
  ])(
    "shows persisted names in a delegate's live %s steps",
    async (tool, displayName, label, name) => {
      server.use(
        subSession(
          [
            {
              role: "assistant",
              content: "",
              tool_calls: [
                {
                  id: "run-1",
                  display_name: displayName,
                  function: {
                    name: tool,
                    arguments: JSON.stringify(
                      tool === "run_agent"
                        ? { library_agent_id: "library-id" }
                        : tool === "run_block"
                          ? { block_id: "block-id", input_data: {} }
                          : { review_id: "review-id" },
                    ),
                  },
                },
              ],
            },
          ],
          { chat_status: "running", active_stream: null },
        ),
      );
      render(
        <SubSessionCard
          output={{ status: "running", sub_session_id: "sub-1" }}
        />,
      );
      expect(await screen.findByText(`${label} "${name}"…`)).toBeDefined();
    },
  );

  it("matches historical delegate output names by tool call ID", async () => {
    server.use(
      subSession(
        [
          {
            role: "assistant",
            content: "",
            tool_calls: [
              { id: "run-1", function: { name: "run_agent", arguments: "{}" } },
              { id: "run-2", function: { name: "run_agent", arguments: "{}" } },
            ],
          },
          {
            role: "tool",
            tool_call_id: "run-2",
            content: '{"agent_name":"Second workflow"}',
            tool_calls: null,
          },
          {
            role: "tool",
            tool_call_id: "run-1",
            content: '{"graph_name":"First workflow"}',
            tool_calls: null,
          },
        ],
        { chat_status: "idle", active_stream: null },
      ),
    );
    render(
      <SubSessionCard
        output={{ status: "running", sub_session_id: "sub-1" }}
      />,
    );
    expect(await screen.findByText('Ran agent "First workflow"')).toBeDefined();
    expect(screen.getByText('Ran agent "Second workflow"')).toBeDefined();
  });

  it("streams the delegate's recent tools and latest words while running", async () => {
    server.use(
      subSession([
        {
          role: "assistant",
          content: "",
          tool_calls: [
            { id: "t1", function: { name: "find_agent", arguments: "{}" } },
          ],
        },
        {
          role: "assistant",
          content: "Looking for the right agent now.",
          tool_calls: null,
        },
      ]),
    );

    render(
      <SubSessionCard
        output={{
          status: "running",
          sub_session_id: "sub-1",
          sub_autopilot_session_link: "/copilot?sessionId=sub-1",
        }}
      />,
    );

    expect(
      await screen.findByText("Looking for the right agent now."),
    ).toBeDefined();
    expect(screen.getByLabelText("Open sub-session").getAttribute("href")).toBe(
      "/copilot?sessionId=sub-1",
    );
  });

  it("scopes the live view to the current turn on a reused sub-session", async () => {
    server.use(
      subSession([
        { role: "user", content: "Build the discord clone", tool_calls: null },
        {
          role: "assistant",
          content: "Done. Discord clone is built.",
          tool_calls: null,
        },
        { role: "user", content: "Now add dark mode", tool_calls: null },
        {
          role: "assistant",
          content: "Starting on dark mode.",
          tool_calls: [
            { id: "t2", function: { name: "find_agent", arguments: "{}" } },
          ],
        },
      ]),
    );

    render(
      <SubSessionCard
        output={{ status: "running", sub_session_id: "sub-1" }}
      />,
    );

    expect(await screen.findByText("Starting on dark mode.")).toBeDefined();
    expect(screen.queryByText("Done. Discord clone is built.")).toBeNull();
  });

  it("renders no card under a delegate row — the wire node and status line own it", () => {
    const { container } = render(
      <ToolResult
        row={{
          key: "delegate",
          category: "agent",
          text: "Handing off to a teammate",
          state: "running",
          tool: "delegate_to_expert",
          input: { expert_id: "exp-1", prompt: "Create a chat app" },
        }}
      />,
    );

    expect(container.textContent).toBe("");
  });

  it("renders no card under a finished delegate either", () => {
    const { container } = render(
      <ToolResult
        row={{
          key: "delegate",
          category: "agent",
          text: "Teammate handled it",
          state: "done",
          tool: "delegate_to_expert",
          input: {},
          output: {
            status: "completed",
            response: "Here is the full brief the teammate wrote.",
            sub_session_id: "sub-1",
            expert: { name: "Vera", role: "Research" },
          },
        }}
      />,
    );

    expect(container.textContent).toBe("");
  });

  /** The delegate/handoff tools name themselves, but a result poll is the
   *  same tool for the model's own scratch sub and for a teammate's thread —
   *  the `expert` on the output is the only thing telling them apart. */
  it("renders no card for a result poll of a teammate's run", () => {
    const { container } = render(
      <ToolResult
        row={{
          key: "poll",
          category: "agent",
          text: "Checking on the teammate",
          state: "done",
          tool: "get_sub_session_result",
          input: {},
          output: {
            status: "running",
            sub_session_id: "sub-1",
            expert: { name: "Vera", role: "Research" },
          },
        }}
      />,
    );

    expect(container.textContent).toBe("");
  });

  /** A poll is capped after 5 minutes so a forgotten tab stops hammering
   *  the API. A long run reaches that cap while it is genuinely still
   *  working, so the pill must stop claiming to know. */
  it("stops asserting running once the poll cap expires", async () => {
    server.use(subSession([], { chat_status: "running", active_stream: null }));
    vi.useFakeTimers({ shouldAdvanceTime: true });

    try {
      render(
        <SubSessionCard
          output={{ status: "running", sub_session_id: "sub-1" }}
        />,
      );

      await vi.advanceTimersByTimeAsync(POLL_CAP_MS + 1);

      expect(screen.getByText("unknown")).toBeDefined();
      expect(screen.queryByText("running")).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it("finds an expert's running session behind a wall of pinned ones", async () => {
    server.use(expertSessions());

    render(
      <SubSessionPendingCard
        input={{ expert_id: "exp-1", prompt: "Create a chat app" }}
      />,
    );

    expect(
      (
        await screen.findByRole("link", { name: "Open sub-session" })
      ).getAttribute("href"),
    ).toBe("/copilot?sessionId=sub-live");
  });

  it("stays quiet once the sub-session has finished", () => {
    render(
      <SubSessionCard
        output={{
          status: "completed",
          response: "All done",
          sub_session_id: "sub-1",
        }}
      />,
    );

    expect(screen.getByText("All done")).toBeDefined();
    expect(screen.queryByText("Looking for the right agent now.")).toBeNull();
  });

  it("says live updates failed instead of spinning on running forever", async () => {
    server.use(failingSubSession());

    render(
      <SubSessionCard
        output={{ status: "running", sub_session_id: "sub-1" }}
      />,
    );

    expect(await screen.findByText("Couldn't load live updates")).toBeDefined();
    expect(screen.queryByText("running")).toBeNull();
    expect(screen.getByText("unknown")).toBeDefined();
    expect(
      screen
        .getByRole("link", { name: "Open sub-session" })
        .getAttribute("href"),
    ).toBe("/copilot?sessionId=sub-1");
  });

  it("drops the running pill on the pending card when the poll fails", async () => {
    server.use(failingSubSession());

    render(
      <SubSessionPendingCard
        input={{ sub_session_id: "sub-1", prompt: "Create a chat app" }}
      />,
    );

    expect(await screen.findByText("unknown")).toBeDefined();
    expect(screen.queryByText("running")).toBeNull();
  });
});
