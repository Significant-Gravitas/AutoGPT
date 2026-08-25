import { getGetV2GetSessionMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it } from "vitest";
import { SubSessionCard } from "../AgentCards";
import { ToolResult } from "../ToolResult";

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

  it("shows who is on it — and nothing else — before a blocking delegate returns", () => {
    render(
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

    expect(screen.getByText("Sub-AutoPilot")).toBeDefined();
    expect(screen.getByText("running")).toBeDefined();
    // A teammate's thread is their own workspace: the delegated card is
    // status-only, so the prompt preview stays out of the parent chain.
    expect(screen.queryByText("Create a chat app")).toBeNull();
  });

  it("keeps a finished delegate card to name, role, status, time, and link", () => {
    render(
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
            sub_autopilot_session_link: "/copilot?sessionId=sub-1",
            elapsed_seconds: 75,
            expert: { name: "Vera", role: "Research" },
          },
        }}
      />,
    );

    expect(screen.getByText(/Vera/)).toBeDefined();
    expect(screen.getByText("Research")).toBeDefined();
    expect(screen.getByText("completed")).toBeDefined();
    expect(screen.getByText("1m 15s")).toBeDefined();
    expect(
      screen
        .getByRole("link", { name: "Open sub-session" })
        .getAttribute("href"),
    ).toBe("/copilot?sessionId=sub-1");
    expect(
      screen.queryByText("Here is the full brief the teammate wrote."),
    ).toBeNull();
  });

  /** The delegate/handoff tools name themselves, but a result poll is the
   *  same tool for the model's own scratch sub and for a teammate's thread —
   *  the `expert` on the output is the only thing telling them apart. */
  it("keeps a delegated result poll minimal and still flips its pill when the teammate lands", async () => {
    server.use(
      subSession(
        [
          {
            role: "assistant",
            content: "Looking for the right agent now.",
            tool_calls: null,
          },
        ],
        { chat_status: "idle", active_stream: null },
      ),
    );

    render(
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
            response: "Half of the brief so far.",
            sub_session_id: "sub-1",
            sub_autopilot_session_link: "/copilot?sessionId=sub-1",
            expert: { name: "Vera", role: "Research" },
          },
        }}
      />,
    );

    // The pill can only flip off the frozen "running" once the poll this hook
    // owns has answered — so this also proves the card polls with no live
    // view mounted to do it for them.
    expect(await screen.findByText("completed")).toBeDefined();
    expect(screen.queryByText("Looking for the right agent now.")).toBeNull();
    expect(screen.queryByText("Half of the brief so far.")).toBeNull();
  });

  it("finds the delegate's running session behind a wall of pinned ones", async () => {
    server.use(expertSessions());

    render(
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
      <ToolResult
        row={{
          key: "delegate",
          category: "agent",
          text: "Handing off to a teammate",
          state: "running",
          tool: "delegate_to_expert",
          input: { sub_session_id: "sub-1", prompt: "Create a chat app" },
        }}
      />,
    );

    expect(await screen.findByText("unknown")).toBeDefined();
    expect(screen.queryByText("running")).toBeNull();
  });
});
